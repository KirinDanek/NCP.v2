"""
run_watermark_pruning_experiment.py

Tests whether NCP can ablate spurious hanzi-watermark subspaces from VGG16's
decision strategy for a binary carton/dugong or crate/packet ImageNet task.

Experiment design
-----------------
For each (experiment, pruner, seed) combination:

  1. Build dataset with on-the-fly synthetic watermarks:
       - Val:    first 150 pos + 150 neg, each in WM and NWM form → 600 entries
                 → used as the eval set during hyperparameter sweeps
       - Train:  next 650 pos (325 WM + 325 NWM) + 650 neg (all NWM)
                 → watermark is a spurious cue correlated only with positive class
       - Rank:   positive-only: 500 pos; full-loader: 250 pos + 250 neg
                 → used exclusively for LRP filter-importance scoring
       - Test:   remaining images, each in WM and NWM form (held out until final eval)
                 → evaluates all four subgroups c1w1/c1w0/c0w1/c0w0

  2. Warm-start classifier head (up to 15 epochs, early stopping).

  3. Run NCP or vanilla iterative prune-fine-tune loop.
       NCP: ablates `--spurious_subspace` from the DRSA projection matrix U.
       Vanilla: standard LRP-based filter pruning on plain VGG16.

  4. Save per-iteration subgroup + overall statistics to
       {out_dir}/{experiment}/{pruner}/seed{seed}/stats.pt
     (no model weights saved unless --save_model is set).

Usage
-----
# Single run:
python run_watermark_pruning_experiment.py \\
    --experiment carton \\
    --pruner ncp \\
    --spurious_subspace 2 \\
    --seed 0 \\
    --out_dir /n/fs/ncp/NCP.v2/results/watermark_experiment/

# SLURM array: see watermark_pruning.slurm
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, '/n/fs/ncp/NCP.v2')
sys.path.insert(0, '/n/fs/ncp/watermark_imagenet')

from sklearn.metrics import average_precision_score
from AugmentedVGG16 import ablate_subspace_matrix, AugmentedVGG16
from prune_vgg import PruningFineTuner, VanillaVGGAdapter, AugmentedVGGAdapter
from watermark_transform import AddWatermark
from lrp import lrp as lrp_fn


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

#   pos_dir  — directory containing positive-class images (unwatermarked originals)
#   neg_dir  — directory containing negative-class images
#   u_path   — DRSA projection matrix (.pt)
EXPERIMENT_CONFIGS = {
    'carton': {
        'pos_dir': '/n/fs/ncp/NCP.v2/data/images/carton_dugong/original/n02971356',
        'neg_dir': '/n/fs/ncp/NCP.v2/data/images/carton_dugong/original/n02074367',
        'u_path':  '/n/fs/ncp/NCP.v2/data/projection_matrices/U_carton_tensor.pt',
    },
    'crate': {
        # n03127925_binary/target: 1113 original full-size crate images
        'pos_dir': '/n/fs/ncp/NCP.v2/data/images/crate_packet/imagenet_n03127925_binary/target',
        'neg_dir': '/n/fs/ncp/NCP.v2/data/images/crate_packet/imagenet_crate_packet_original/n03871628',
        'u_path':  '/n/fs/ncp/NCP.v2/data/projection_matrices/U_crate_tensor.pt',
    },
}

SUBSPACE_DIMS = [128, 128, 128, 128]   # 4 subspaces × 128 dims each

# Split sizes
# Layout (per class, sorted order):
#   [0 : N_VAL]                        → validation
#   [N_VAL : N_VAL+N_TRAIN_POS/NEG]    → train / rank pool
#   [N_VAL+N_TRAIN_POS/NEG : ...]      → test  (same boundary as before: index 800)
N_VAL       = 150    # validation images per class (WM+NWM copies → 600 val entries total)
N_TRAIN_POS = 650    # positive-class images for fine-tuning  (was 800; 150 moved to val)
N_TRAIN_NEG = 650    # negative-class images for fine-tuning
N_RANK_POS  = 500    # positive-only: all 500 pos; full-loader: first 250 pos
N_RANK_NEG  = 250    # full-loader only: 250 neg → 250+250=500 total (matches pos-only)

IMAGE_EXTS = {'.jpg', '.jpeg', '.JPEG', '.png', '.PNG', '.JPEG'}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WatermarkDataset(Dataset):
    """PIL images with optional on-the-fly watermarking. Yields (tensor, cls, wm)."""

    def __init__(self, samples, rc_transform, tensor_transform, add_watermark_fn):
        """
        samples          : list of (path_str, class_label, watermark_label)
        rc_transform     : Resize+CenterCrop → 224×224 PIL
        tensor_transform : ToTensor+Normalize
        add_watermark_fn : AddWatermark instance (operates on 224×224 PIL)
        """
        self.samples = samples
        self.rc      = rc_transform
        self.tt      = tensor_transform
        self.wm_fn   = add_watermark_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls, wm = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.rc(img)               # → 224×224 PIL
        if wm:
            img = self.wm_fn(img)        # AddWatermark on PIL → PIL
        img = self.tt(img)               # → float tensor
        return img, int(cls), int(wm)


class NaturalTestDataset(Dataset):
    """Like WatermarkDataset but never applies a synthetic watermark.
    WM labels come from the sample tuple (manually annotated natural watermarks)."""

    def __init__(self, samples, rc_transform, tensor_transform):
        self.samples = samples
        self.rc      = rc_transform
        self.tt      = tensor_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls, wm = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.rc(img)
        img = self.tt(img)
        return img, int(cls), int(wm)


# ---------------------------------------------------------------------------
# Data-split helpers
# ---------------------------------------------------------------------------

def _list_images(directory):
    """Return sorted list of image file paths in a directory."""
    p = Path(directory)
    return sorted(str(f) for f in p.iterdir() if f.suffix in IMAGE_EXTS)


def build_splits(pos_dir, neg_dir,
                 n_val=N_VAL,
                 n_train_pos=N_TRAIN_POS,
                 n_train_neg=N_TRAIN_NEG,
                 n_rank_pos=N_RANK_POS,
                 n_rank_neg=N_RANK_NEG):
    """
    Build deterministic val / train / rank / test sample lists.

    Layout (per class, sorted order):
      [0 : n_val]                           → validation (WM=0 and WM=1 copies)
      [n_val : n_val+n_train_pos/neg]       → train / rank pool
      [n_val+n_train_pos/neg : ...]         → test (WM=0 and WM=1 copies)

    Val set    : first n_val pos + first n_val neg, each with WM=0 and WM=1 copies
                 → 4 × n_val entries across all 4 subgroups.
    Train set  : next n_train_pos pos (50 % WM by index parity)
                 + next n_train_neg neg (all NWM).
    Rank sets  : rank_pos_samples — first n_rank_pos positives from train (same WM parity).
                 rank_neg_samples — first n_rank_neg negatives from train (all NWM).
                 positive-only loader uses rank_pos (500 images).
                 full loader uses rank_pos[:250] + rank_neg (250+250=500 images).
    Test set   : remaining images, each with WM=0 and WM=1 copies.

    Returns
    -------
    train_samples, rank_pos_samples, rank_neg_samples, val_samples, test_samples
    """
    pos_files = _list_images(pos_dir)
    neg_files = _list_images(neg_dir)

    n_val       = min(n_val, len(pos_files), len(neg_files))
    n_train_pos = min(n_train_pos, len(pos_files) - n_val)
    n_train_neg = min(n_train_neg, len(neg_files) - n_val)
    n_rank_pos  = min(n_rank_pos, n_train_pos)
    n_rank_neg  = min(n_rank_neg, n_train_neg)

    val_pos   = pos_files[:n_val]
    val_neg   = neg_files[:n_val]
    train_pos = pos_files[n_val : n_val + n_train_pos]
    train_neg = neg_files[n_val : n_val + n_train_neg]
    test_pos  = pos_files[n_val + n_train_pos:]
    test_neg  = neg_files[n_val + n_train_neg:]

    # Val: both WM=0 and WM=1 copies for all 4 subgroups
    val_samples = (
        [(f, 1, 0) for f in val_pos] + [(f, 1, 1) for f in val_pos]
        + [(f, 0, 0) for f in val_neg] + [(f, 0, 1) for f in val_neg]
    )

    # Train: even-index positive → WM=1; odd-index → WM=0  (deterministic 50/50)
    train_samples = (
        [(f, 1, i % 2) for i, f in enumerate(train_pos)]
        + [(f, 0, 0) for f in train_neg]
    )

    # Rank pos: first n_rank_pos positives from train pool (same WM parity)
    rank_pos_samples = [(f, 1, i % 2) for i, f in enumerate(train_pos[:n_rank_pos])]

    # Rank neg: first n_rank_neg negatives from train pool (all NWM)
    rank_neg_samples = [(f, 0, 0) for f in train_neg[:n_rank_neg]]

    # Test: balance by class, then include WM=0 and WM=1 copies
    n_test   = min(len(test_pos), len(test_neg))
    test_pos = test_pos[:n_test]
    test_neg = test_neg[:n_test]
    test_samples = (
        [(f, 1, 0) for f in test_pos] + [(f, 1, 1) for f in test_pos]
        + [(f, 0, 0) for f in test_neg] + [(f, 0, 1) for f in test_neg]
    )

    n_pos_wm  = sum(wm for _, cls, wm in train_samples if cls == 1)
    n_pos_nwm = sum(1 - wm for _, cls, wm in train_samples if cls == 1)
    print(f"[data] val   : {len(val_samples):4d}  "
          f"({n_val} pos × 2 + {n_val} neg × 2)")
    print(f"[data] train : {len(train_samples):4d}  "
          f"({len(train_pos)} pos [{n_pos_wm} WM + {n_pos_nwm} NWM] "
          f"+ {len(train_neg)} neg NWM)")
    print(f"[data] rank  : {len(rank_pos_samples):4d} pos  "
          f"(full-loader uses first 250 pos + {len(rank_neg_samples)} neg)")
    print(f"[data] test  : {len(test_samples):4d}  "
          f"({n_test} pos × 2 + {n_test} neg × 2, balanced)")

    return train_samples, rank_pos_samples, rank_neg_samples, val_samples, test_samples


def _load_natural_wm_indices(txt_path):
    """Return a set of integer file-indices from test-indices.txt (skip comment lines)."""
    indices = set()
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            indices.add(int(line))
    return indices


def _index_from_path(path):
    """Extract integer index from filenames like n02971356_00812.jpeg → 812."""
    return int(Path(path).stem.split('_')[-1])


def build_natural_test_samples(pos_dir, neg_dir, test_indices_path, n_test=400):
    """
    Build a balanced test sample list using *natural* watermark labels.

    Carton (pos): images at sorted positions [800 : 800+n_test].
      WM=1 if the filename index appears in test_indices_path, else WM=0.
    Dugong (neg): images at sorted positions [800 : 800+n_test].
      WM=0 for all (dugong has no natural watermarks).

    n_test is capped by the available test images for either class.
    """
    wm_set  = _load_natural_wm_indices(test_indices_path)
    pos_all = _list_images(pos_dir)
    neg_all = _list_images(neg_dir)

    n_test = min(n_test, len(pos_all) - 800, len(neg_all) - 800)

    samples = []
    for path in pos_all[800 : 800 + n_test]:
        wm = 1 if _index_from_path(path) in wm_set else 0
        samples.append((path, 1, wm))
    for path in neg_all[800 : 800 + n_test]:
        samples.append((path, 0, 0))

    n_wm = sum(w for _, _, w in samples)
    print(f"[natural test] {len(samples)} samples  "
          f"(pos={n_test}, neg={n_test}, wm={n_wm}/{n_test} pos images)")
    return samples


def make_loaders(train_samples, rank_pos_samples, rank_neg_samples,
                 val_samples, test_samples,
                 add_wm, batch_size, cuda, rank_loader_type='positive_only'):
    """Construct train / rank / val / test DataLoaders from sample lists.

    rank_loader_type:
      'positive_only' — rank loader uses all rank_pos_samples (500 pos).
      'full_loader'   — rank loader uses rank_pos_samples[:250] + rank_neg_samples
                        (250 pos + 250 neg = 500 total).
    """
    rc = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    tt = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    worker_kwargs = (
        {'num_workers': 3, 'pin_memory': True, 'multiprocessing_context': 'spawn',
         'persistent_workers': True}
        if cuda else {}
    )

    if rank_loader_type == 'full_loader':
        rank_samples = rank_pos_samples[:250] + rank_neg_samples
    else:
        rank_samples = rank_pos_samples

    train_ds = WatermarkDataset(train_samples,  rc, tt, add_wm)
    rank_ds  = WatermarkDataset(rank_samples,   rc, tt, add_wm)
    val_ds   = WatermarkDataset(val_samples,    rc, tt, add_wm)
    test_ds  = WatermarkDataset(test_samples,   rc, tt, add_wm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **worker_kwargs)
    rank_loader  = DataLoader(rank_ds,  batch_size=batch_size, shuffle=False, **worker_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **worker_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **worker_kwargs)

    return train_loader, rank_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# PruningFineTuner subclass
# ---------------------------------------------------------------------------

class WatermarkPruningFineTuner(PruningFineTuner):
    """
    Pruning fine-tuner for the watermark spurious-cue experiment.

    Differences from PruningFineTuner:
      - setup_dataloaders uses pre-built WatermarkDataset loaders instead of
        calling data.py, so no changes to data.py are needed.
      - self.test_loader is set to val_loader during the hyperparameter sweep so
        that PruningFineTuner.test() evaluates on val; held-out test data lives in
        self._held_out_test_loader and is not touched during the sweep.
      - get_candidates_to_prune temporarily swaps to a dedicated rank_loader
        for LRP filter scoring (500 pos or 250 pos+250 neg), then restores
        train_loader for subsequent fine-tuning. This mirrors the
        BasketballPruningFineTuner pattern.
    """

    def __init__(self, args, model, adapter,
                 train_samples, rank_pos_samples, rank_neg_samples,
                 val_samples, test_samples, add_wm,
                 natural_test_samples=None, subspace_dims=None, U_orig=None):
        # Store data BEFORE calling super().__init__(), because super().__init__()
        # calls self.setup_dataloaders() via Python MRO.
        self._wm_train_samples       = train_samples
        self._wm_rank_pos_samples    = rank_pos_samples
        self._wm_rank_neg_samples    = rank_neg_samples
        self._wm_val_samples         = val_samples
        self._wm_test_samples        = test_samples
        self._add_wm                 = add_wm
        self._natural_test_samples   = natural_test_samples
        self._subspace_dims          = subspace_dims or SUBSPACE_DIMS
        # Original (non-ablated) U matrix for subspace relevance projection.
        # Shape: (512, 512), stored as float32 CPU tensor, moved to device on use.
        self._U_orig                 = U_orig
        self.subspace_relevance_tot  = []
        super().__init__(args, model, adapter)

    # ------------------------------------------------------------------
    # Dataloader setup
    # ------------------------------------------------------------------

    def setup_dataloaders(self):
        train_loader, rank_loader, val_loader, test_loader = make_loaders(
            self._wm_train_samples,
            self._wm_rank_pos_samples,
            self._wm_rank_neg_samples,
            self._wm_val_samples,
            self._wm_test_samples,
            self._add_wm,
            batch_size=self.args.train_batch_size,
            cuda=self.args.cuda,
            rank_loader_type=getattr(self.args, 'rank_loader_type', 'positive_only'),
        )
        self.train_loader = train_loader
        self._rank_loader = rank_loader

        if self._natural_test_samples is not None:
            # Replace the eval loader with the natural-watermark test set.
            rc = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
            tt = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            worker_kw = (
                {'num_workers': 3, 'pin_memory': True,
                 'multiprocessing_context': 'spawn', 'persistent_workers': True}
                if self.args.cuda else {}
            )
            nat_ds = NaturalTestDataset(self._natural_test_samples, rc, tt)
            self.test_loader = DataLoader(
                nat_ds, batch_size=self.args.test_batch_size,
                shuffle=False, **worker_kw)
        else:
            # Default: evaluate on val during sweeps, held-out test with --eval_on_test.
            use_test = getattr(self.args, 'eval_on_test', False)
            self.test_loader = test_loader if use_test else val_loader

        self.train_num    = len(train_loader)
        self.test_num     = len(self.test_loader)
        # Required by PruningFineTuner (used in run_PFT.py log, empty here)
        self.eval_supplement_fnames = []

    # ------------------------------------------------------------------
    # LRP scoring: use rank_loader (positive-class only)
    # ------------------------------------------------------------------

    def get_candidates_to_prune(self, num_filters_to_prune):
        _saved_loader = self.train_loader
        _saved_num    = self.train_num
        _saved_rank_n = getattr(self.args, 'rank_n', None)
        # Disable the rank_n cap: rank_loader is already bounded to n_rank images.
        self.args.rank_n  = None
        self.train_loader = self._rank_loader
        self.train_num    = len(self._rank_loader)
        try:
            result = super().get_candidates_to_prune(num_filters_to_prune)
        finally:
            self.train_loader = _saved_loader
            self.train_num    = _saved_num
            self.args.rank_n  = _saved_rank_n
        return result

    # ------------------------------------------------------------------
    # Per-subspace LRP relevance and AP (NCP only, natural test set)
    # ------------------------------------------------------------------

    def _compute_subspace_relevances(self):
        """
        Compute per-subspace LRP relevance on self.test_loader, projecting
        through the ORIGINAL (non-ablated) U matrix rather than the ablated
        encode/decode.  This gives a meaningful signal across pruning iterations:
        as NCP removes filters associated with the ablated subspace, the relevance
        attributable to that subspace via the original U should decrease.

        Method:
          1. Run LRP with augmented=False (already the state during test()).
          2. Capture relevance R at conv4_3 output (after LRP through pool4).
          3. Project: projected = einsum('bchw,cd->bdhw', R, U_orig).
             Subspace s occupies projected channels [s*sz : (s+1)*sz].
          4. Compute per-subspace sums, group means, and AP for WM and class.

        Returns a dict with keys:
          niter, mean_all, mean_wm0, mean_wm1,
          ap_wm (K,), ap_class (K,)
        or None if U_orig is not set or test_loader is absent.
        """
        if self._U_orig is None or not hasattr(self, 'test_loader'):
            return None

        self.model.eval()

        # Module list with augmented=False (current state during test())
        module_list = (list(self.model.before)
                       + list(self.model.after)
                       + list(self.model.classifier))
        # after[0] = pool4 (MaxPool); its forward index in module_list
        after0_fwd_idx = len(list(self.model.before))
        rev_mods = list(reversed(module_list))
        total    = len(module_list)

        # LRP rules (alpha-1 beta-0); positional, fixed throughout pruning
        rules = []
        for rev_idx, mod in enumerate(rev_mods):
            lrp_var, param = self.adapter.get_backward_lrp_rule(
                mod, rev_idx, total, 'alphabeta', 1.0)
            rules.append((lrp_var, param))

        K  = len(self._subspace_dims)
        sz = self._subspace_dims[0]   # 128 for standard DRSA
        U  = self._U_orig.to(next(self.model.parameters()).device)  # (512, 512)

        accum      = [[] for _ in range(K)]
        wm_all     = []
        cls_all    = []

        for batch in self.test_loader:
            x, cls, wm = batch
            if self.args.cuda:
                x = x.cuda(non_blocking=True)

            # Forward with hooks to capture .input / .output on each module
            handles = []
            for mod in module_list:
                def _hook(m, inp, out, _m=mod):
                    _m.input  = inp[0]
                    _m.output = out.data
                handles.append(mod.register_forward_hook(_hook))
            try:
                with torch.no_grad():
                    logits = self.model(x)
            finally:
                for h in handles:
                    h.remove()

            # LRP: seed relevance from positive-class logit
            R = torch.zeros_like(logits)
            R[:, 1] = logits[:, 1]

            conv43_R = None
            for mod_idx, (mod, (lrp_var, param)) in enumerate(zip(rev_mods, rules)):
                R = lrp_fn(mod, R, lrp_var=lrp_var, param=param)
                # Capture R immediately after LRP through after[0] (pool4).
                # At this point R = relevance at pool4 INPUT = conv4_3 output.
                if (len(module_list) - 1 - mod_idx) == after0_fwd_idx:
                    conv43_R = R.clone()

            if conv43_R is None:
                continue

            # Project onto original U: projected[b,d,h,w] = sum_c U[c,d]*R[b,c,h,w]
            # Shape: (B, 512, H, W)
            projected = torch.einsum('bchw,cd->bdhw', conv43_R, U)

            for si in range(K):
                score = projected[:, si*sz:(si+1)*sz, :, :].sum(dim=(1, 2, 3))
                accum[si].append(score.cpu().float().numpy())

            wm_all.append(wm.numpy())
            cls_all.append(cls.numpy())

        per_sub   = [np.concatenate(accum[si]) for si in range(K)]
        wm_labels = np.concatenate(wm_all)
        cls_labels = np.concatenate(cls_all)

        def _ap_safe(y, scores):
            if y.sum() == 0 or y.sum() == len(y):
                return float('nan')
            return float(average_precision_score(y, scores))

        result = {
            'niter':     self.niter,
            'mean_all':  np.array([s.mean() for s in per_sub]),
            'ap_wm':     np.array([_ap_safe(wm_labels,  s) for s in per_sub]),
            'ap_class':  np.array([_ap_safe(cls_labels, s) for s in per_sub]),
        }
        for g in [0, 1]:
            mask_g = (wm_labels == g)
            if mask_g.sum() > 0:
                result[f'mean_wm{g}'] = np.array(
                    [s[mask_g].mean() for s in per_sub])

        self.model.train()
        return result

    def test(self, compute_subgroup=False):
        super().test(compute_subgroup=compute_subgroup)
        result = self._compute_subspace_relevances()
        if result is not None:
            self.subspace_relevance_tot.append(result)

    def prune(self, **kwargs):
        self.subspace_relevance_tot = []
        return super().prune(**kwargs)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(args, u_path):
    if args.pruner == 'ncp':
        U = torch.load(u_path, map_location='cpu', weights_only=False).float()
        U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, list(args.spurious_subspace))
        model = AugmentedVGG16(U_ab, U_ab_T)
    else:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    model.classifier[6] = nn.Linear(4096, 2)

    if args.cuda:
        model = model.cuda()
    return model


# ---------------------------------------------------------------------------
# Warmup: train classifier head with early stopping
# ---------------------------------------------------------------------------

def warmup_head(model, train_loader, args, criterion):
    """Train classifier[6] only, for exactly warmup_epochs on the (biased) train set.

    No early stopping: the test set is reserved for pure evaluation only.
    The augmented path is bypassed during warmup so the head learns from
    raw conv features, consistent with run_PFT.py.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier[6].parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=args.lr)

    _was_augmented = getattr(model, 'augmented', False)
    if _was_augmented:
        model.augmented = False

    model.train()
    for epoch in range(args.warmup_epochs):
        running_loss = 0.0
        for bidx, batch in enumerate(train_loader):
            x, y = batch[0], batch[1]
            if args.cuda:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if bidx % 20 == 0:
                print(f"  warmup e{epoch+1} b{bidx} loss={loss.item():.4f}")
        print(f"  warmup epoch {epoch+1}/{args.warmup_epochs}  "
              f"avg_loss={running_loss/len(train_loader):.4f}")

    if _was_augmented:
        model.augmented = True


# ---------------------------------------------------------------------------
# Single-run pipeline
# ---------------------------------------------------------------------------

def run_one(args):
    cfg = EXPERIMENT_CONFIGS[args.experiment]

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Data
    train_samples, rank_pos_samples, rank_neg_samples, val_samples, test_samples = \
        build_splits(cfg['pos_dir'], cfg['neg_dir'])
    add_wm = AddWatermark(image_size=224)

    # Natural watermark test set (replaces synthetic eval when --natural_test is set)
    natural_test_samples = None
    if getattr(args, 'natural_test', False):
        if not getattr(args, 'test_indices_path', None):
            raise ValueError('--natural_test requires --test_indices_path')
        natural_test_samples = build_natural_test_samples(
            cfg['pos_dir'], cfg['neg_dir'],
            args.test_indices_path,
            n_test=getattr(args, 'n_natural_test', 400),
        )

    # Model
    model   = build_model(args, cfg['u_path'])
    adapter = (AugmentedVGGAdapter(model) if args.pruner == 'ncp'
               else VanillaVGGAdapter(model))

    # Original (non-ablated) U matrix for per-iteration subspace relevance.
    # Loaded separately from the ablated model so the ablation does not affect
    # the projection used to evaluate subspace activity.
    U_orig = None
    if natural_test_samples is not None:
        U_orig = torch.load(cfg['u_path'], map_location='cpu',
                            weights_only=False).float()

    # Tuner (calls setup_dataloaders internally)
    tuner = WatermarkPruningFineTuner(
        args, model, adapter,
        train_samples, rank_pos_samples, rank_neg_samples,
        val_samples, test_samples, add_wm,
        natural_test_samples=natural_test_samples,
        subspace_dims=list(args.subspace_dims),
        U_orig=U_orig,
    )

    # Warmup: train classifier head on the biased training set only.
    # Test loader is reserved for pure evaluation — not used here.
    print("=== Warmup ===")
    criterion = nn.CrossEntropyLoss()
    warmup_head(model, tuner.train_loader, args, criterion)

    # Re-enable all params for the pruning phase
    for p in model.parameters():
        p.requires_grad = True

    # Prune
    print("=== Pruning ===")
    tuner.prune(
        fine_tune_conv_layers=args.fine_tune_conv_layers,
        fine_tune_without_augmented_layers=args.fine_tune_without_augmented_layers,
        final_finetune_epochs=args.final_finetune_epochs,
        iter_finetune_epochs=args.iter_finetune_epochs,
    )

    # Save stats (no model weights)
    out_dir = Path(args.out_dir) / args.experiment / args.pruner / f'seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'stats.pt'

    torch.save({
        'experiment':            args.experiment,
        'pruner':                args.pruner,
        'seed':                  args.seed,
        'spurious_subspace':     args.spurious_subspace,
        'iter_finetune_epochs':  args.iter_finetune_epochs,
        'rank_loader_type':      args.rank_loader_type,
        'eval_on_test':          args.eval_on_test,
        'n_val':                 N_VAL,
        'n_train_pos':           N_TRAIN_POS,
        'n_train_neg':           N_TRAIN_NEG,
        'n_rank_pos':            N_RANK_POS,
        'n_rank_neg':            N_RANK_NEG,
        'train_loss':            tuner.train_loss_tot,
        'train_acc':             tuner.train_acc_tot,
        # eval_* keys reflect the loader used: val (sweep) or test (--eval_on_test).
        'eval_loss':             tuner.test_loss_tot,
        'eval_acc':              tuner.test_acc_tot,
        'eval_iter':             tuner.test_iter,
        'eval_precision_per_class': tuner.test_precision_tot,
        'eval_recall_per_class': tuner.test_recall_tot,
        # Each entry is a dict with keys: acc_g{g}y{y}, n_g{g}y{y},
        # precision_g{g}, recall_g{g}, ap_g{g}, ap_overall
        # where g=0 → no-watermark, g=1 → watermark; y=0 → neg, y=1 → pos
        # Subgroup labels: c0w0=(g0y0), c1w0=(g0y1), c0w1=(g1y0), c1w1=(g1y1)
        'eval_subgroup_stats':   tuner.subgroup_stats_tot,
        # Per-iteration subspace LRP relevance (NCP only, natural test set).
        # Each entry: {niter, mean_all (K,), mean_wm0 (K,), mean_wm1 (K,)}
        'subspace_relevance':    tuner.subspace_relevance_tot,
    }, out_path)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description='NCP vs. vanilla pruning on synthetic watermark spurious-cue task.')

    p.add_argument('--experiment', choices=['carton', 'crate'], required=True,
                   help='Which binary classification task to run.')
    p.add_argument('--pruner', choices=['ncp', 'vanilla'], required=True)
    p.add_argument('--seed', type=int, default=0)

    # Subspace to ablate (NCP only).  Identify by inspecting DRSA subspace heatmaps.
    p.add_argument('--spurious_subspace', type=int, nargs='+', default=[0],
                   help='One or more 0-indexed DRSA subspaces to ablate (NCP only). '
                        'E.g. --spurious_subspace 1 3 ablates both subspaces 1 and 3.')

    # Pruning hyperparameters (same defaults as run_PFT.py)
    p.add_argument('--pr_step',   type=float, default=0.05,
                   help='Fraction of filters pruned per iteration.')
    p.add_argument('--total_pr',  type=float, default=0.80,
                   help='Total fraction of filters to prune.')
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--momentum',  type=float, default=0.9)
    p.add_argument('--train_batch_size', type=int, default=32)
    p.add_argument('--test_batch_size',  type=int, default=32)
    p.add_argument('--rank_n',    type=int,   default=None,
                   help='Rank-loader sample cap (None = use all rank images).')
    p.add_argument('--final_finetune_epochs', type=int, default=5)
    p.add_argument('--iter_finetune_epochs',  type=int, default=2,
                   help='Fine-tuning epochs after each pruning iteration.')
    p.add_argument('--rank_loader_type', choices=['positive_only', 'full_loader'],
                   default='positive_only',
                   help='positive_only: 500 pos images for LRP scoring. '
                        'full_loader: 250 pos + 250 neg (same total).')
    p.add_argument('--eval_on_test', action='store_true', default=False,
                   help='Evaluate on the held-out test set instead of val. '
                        'Use only for final paper runs after hyperparameter selection; '
                        'do NOT set during hyperparameter sweeps.')
    p.add_argument('--natural_test', action='store_true', default=False,
                   help='Replace the synthetic-watermark eval set with a natural-watermark '
                        'test set (carton only). Requires --test_indices_path.')
    p.add_argument('--test_indices_path', type=str, default=None,
                   help='Path to test-indices.txt listing carton image indices with natural '
                        'watermarks. Required when --natural_test is set.')
    p.add_argument('--n_natural_test', type=int, default=400,
                   help='Number of test images per class for natural-watermark eval '
                        '(default 400).')

    # Warmup
    p.add_argument('--warmup_epochs', type=int, default=15,
                   help='Fixed number of epochs to train the classifier head before pruning.')

    # Convolution fine-tuning flags
    p.add_argument('--fine_tune_conv_layers', action='store_true', default=True)
    p.add_argument('--no_fine_tune_conv_layers',
                   dest='fine_tune_conv_layers', action='store_false')
    p.add_argument('--fine_tune_without_augmented_layers',
                   action='store_true', default=True)
    p.add_argument('--fine_tune_with_augmented_layers',
                   dest='fine_tune_without_augmented_layers', action='store_false')

    # NCP subspace dims
    p.add_argument('--subspace_dims', type=int, nargs='+', default=SUBSPACE_DIMS)

    # CUDA
    p.add_argument('--cuda',    action='store_true',  default=True)
    p.add_argument('--no_cuda', dest='cuda', action='store_false')

    # Required by PruningFineTuner internals
    p.add_argument('--relevance',    action='store_true', default=True)
    p.add_argument('--no_relevance', dest='relevance', action='store_false')
    p.add_argument('--method_type',  type=str, default='lrp')
    p.add_argument('--min_male_with_attr', type=int, default=0)

    p.add_argument('--out_dir', type=str,
                   default='/n/fs/ncp/NCP.v2/results/watermark_experiment/')

    args = p.parse_args()

    # Required by PruningFineTuner.test() for subgroup stats
    args.data_type = 'watermark_imagenet'

    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()
    print(f"[config] experiment={args.experiment}  pruner={args.pruner}  "
          f"seed={args.seed}  spurious_subspace={args.spurious_subspace}")
    run_one(args)
