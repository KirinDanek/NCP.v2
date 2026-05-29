"""
run_PFT_basketball.py

Basketball-specific NCP pruning script. Differs from run_PFT.py in two ways:

1. Warmup — trains the 2-d classifier head on the full basketball dataset for up
   to 15 epochs, stopping early once validation loss stops improving (patience=3,
   min_delta=1e-3).

2. Positive-class-only LRP ranking — at each pruning iteration the LRP relevance
   pass runs over only the first 500 basketball-class images (sorted by filename),
   so filter importance is scored entirely by how much each filter contributes to
   detecting basketball. Fine-tuning after each prune step still uses the full
   (both-class) training set so the model can recover on all examples.

Usage
-----
python run_PFT_basketball.py \\
    --pruner ncp \\
    --u_filepath /n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt \\
    --irrelevant_subspaces 1 \\
    --out_dir /n/fs/ncp/NCP.v2/results/basketball-ncp/irrelevant_subspace_1
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from pathlib import Path

from AugmentedVGG16 import ablate_subspace_matrix, AugmentedVGG16
from prune_vgg import PruningFineTuner, VanillaVGGAdapter, AugmentedVGGAdapter
from lrp import lrp


# ---------------------------------------------------------------------------
# Helpers (copied from run_PFT.py to avoid its module-level CUDA side-effects)
# ---------------------------------------------------------------------------

def _load_u_matrix(path: str) -> torch.Tensor:
    if path.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 3:
            d, n_sub, sub_dim = arr.shape
            print(f"[U matrix] 3-D array {arr.shape} → reshaping to ({d}, {n_sub * sub_dim})")
            arr = arr.reshape(d, n_sub * sub_dim)
        return torch.tensor(arr, dtype=torch.float32)
    else:
        U = torch.load(path, map_location='cpu')
        if not isinstance(U, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor from {path!r}, got {type(U)}")
        return U.float()


def _load_checkpoint_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        if 'model' in ckpt:
            obj = ckpt['model']
            return obj.state_dict() if hasattr(obj, 'state_dict') else obj
        return ckpt
    if hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()
    raise ValueError(f"Cannot extract state_dict from checkpoint at {path!r}")


def _remap_vgg16_features_to_augmented(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            parts = k.split('.')
            layer_idx = int(parts[1])
            suffix = '.'.join(parts[2:])
            if layer_idx < 23:
                new_sd[f'before.{layer_idx}.{suffix}'] = v
            else:
                new_sd[f'after.{layer_idx - 23}.{suffix}'] = v
        else:
            new_sd[k] = v
    return new_sd


# ---------------------------------------------------------------------------
# Positive-class ranking loader
# ---------------------------------------------------------------------------

def build_basketball_rank_loader(root_dir: str, transform, n: int = 500,
                                  batch_size: int = 32,
                                  num_workers: int = 3,
                                  pin_memory: bool = True) -> DataLoader:
    """Return a DataLoader over the first `n` basketball-class images (by filename).

    Images are drawn directly from the ImageFolder root (not from the random
    train/test split) so the selection is deterministic and independent of the
    80/20 split seed.  Only the positive class ('basketball') is included so that
    LRP relevance scores measure which filters are most important for basketball
    detection, not for the negative class.
    """
    full = datasets.ImageFolder(root_dir, transform=transform)
    basketball_cls = full.class_to_idx.get('basketball')
    if basketball_cls is None:
        raise ValueError(
            f"'basketball' class not found in {root_dir}. "
            f"Available classes: {list(full.class_to_idx.keys())}"
        )

    # Sort samples by filename for determinism, then take first n
    basketball_samples = sorted(
        [(i, path) for i, (path, cls) in enumerate(full.samples) if cls == basketball_cls],
        key=lambda x: x[1],   # sort by path string
    )
    if len(basketball_samples) < n:
        print(f"[rank_loader] Only {len(basketball_samples)} basketball images available "
              f"(requested {n}); using all.")
    indices = [i for i, _ in basketball_samples[:n]]
    rank_dataset = Subset(full, indices)
    print(f"[rank_loader] {len(rank_dataset)} basketball images for LRP ranking "
          f"(class idx={basketball_cls})")

    return DataLoader(
        rank_dataset,
        batch_size=batch_size,
        shuffle=False,   # deterministic ranking
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context='spawn' if pin_memory else None,
    )


# ---------------------------------------------------------------------------
# PruningFineTuner subclass: swap train_loader during ranking only
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inline LRP + heatmap helpers (cached-static columns pre-computed once)
# ---------------------------------------------------------------------------

def _fhook(module, input, output):
    module.input  = input[0]
    module.output = output.data


def _run_lrp(modules_bwd, first_conv, R):
    for m in modules_bwd:
        if m is first_conv:
            R = lrp(m, R, lrp_var='first', param=None)
        else:
            R = lrp(m, R, lrp_var='alpha', param=1)
    return R


def _lrp_vanilla_vgg16(model, x, target_cls):
    feats = list(model.features)
    clf   = list(model.classifier)
    handles = [m.register_forward_hook(_fhook) for m in feats + clf]
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    R = torch.zeros_like(out); R[0, target_cls] = 1.0
    R = _run_lrp(list(reversed(clf)) + list(reversed(feats)), feats[0], R)
    for h in handles: h.remove()
    return R.squeeze(0).cpu().numpy()


def _lrp_pruned_vanilla(model, x, target_cls):
    feats = list(model.features)
    clf   = list(model.classifier)
    handles = [m.register_forward_hook(_fhook) for m in feats + clf]
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    R = torch.zeros_like(out); R[0, target_cls] = 1.0
    R = _run_lrp(list(reversed(clf)) + list(reversed(feats)), feats[0], R)
    for h in handles: h.remove()
    return R.squeeze(0).cpu().numpy()


def _lrp_pruned_ncp(model, x, target_cls):
    before = list(model.before)
    after  = list(model.after)
    clf    = list(model.classifier)
    handles = [m.register_forward_hook(_fhook) for m in before + after + clf]
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    R = torch.zeros_like(out); R[0, target_cls] = 1.0
    first_conv = next(m for m in before if isinstance(m, nn.Conv2d))
    R = _run_lrp(list(reversed(clf)) + list(reversed(after)) + list(reversed(before)),
                 first_conv, R)
    for h in handles: h.remove()
    return R.squeeze(0).cpu().numpy()


def _show_image(ax, pil_img, ylabel=None, title=None):
    import torchvision.transforms as _T
    ax.imshow(_T.Compose([_T.Resize(256), _T.CenterCrop(224)])(pil_img))
    ax.set_xticks([]); ax.set_yticks([])
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title:  ax.set_title(title,   fontsize=9)


def _show_heatmap(ax, hm2d, logit=None, title=None):
    import matplotlib.pyplot as _plt
    from matplotlib.colors import ListedColormap as _LC
    import numpy as _np
    b = _np.abs(hm2d).max() or 1.0
    cmap = _plt.cm.seismic(_np.arange(_plt.cm.seismic.N))
    cmap[:, :3] *= 0.85
    ax.imshow(hm2d, cmap=_LC(cmap), vmin=-b, vmax=b)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)
    xlabel = f"$\\sum R_i$={hm2d.sum():.3f}"
    if logit is not None:
        xlabel += f"  logit={logit:.3f}"
    ax.set_xlabel(xlabel, fontsize=7)


def precompute_static_heatmaps(img_paths, vanilla_imagenet, vanilla_pruned, device):
    """Compute vanilla-imagenet and vanilla-pruned LRP columns once.

    Returns
    -------
    pil_images        : list of PIL images (224×224 crop)
    van_imagenet_hms  : list of 2D heatmaps  (vanilla imagenet)
    van_imagenet_logits : list of floats
    van_pruned_hms    : list of 2D heatmaps  (vanilla pruned, binary basketball logit=0)
    van_pruned_logits : list of floats
    """
    import torchvision.transforms as _T
    from PIL import Image as _Image

    rc = _T.Compose([_T.Resize(256), _T.CenterCrop(224)])
    tt = _T.Compose([
        _T.ToTensor(),
        _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_images, van_im_hms, van_im_logits, van_pr_hms, van_pr_logits = [], [], [], [], []
    for p in img_paths:
        img = _Image.open(p).convert('RGB')
        img_cropped = rc(img)
        pil_images.append(img_cropped)
        x = tt(img_cropped).to(device)

        hm_im = _lrp_vanilla_vgg16(vanilla_imagenet, x, 430).sum(axis=0)
        van_im_logits.append(float(vanilla_imagenet(x.unsqueeze(0)).squeeze()[430]))
        van_im_hms.append(hm_im)

        hm_pr = _lrp_pruned_vanilla(vanilla_pruned, x, 0).sum(axis=0)
        van_pr_logits.append(float(vanilla_pruned(x.unsqueeze(0)).squeeze()[0]))
        van_pr_hms.append(hm_pr)

    print(f"[heatmap] pre-computed static columns for {len(img_paths)} images")
    return pil_images, van_im_hms, van_im_logits, van_pr_hms, van_pr_logits


def generate_heatmap(ncp_model, device, img_paths, pil_images,
                     van_im_hms, van_im_logits, van_pr_hms, van_pr_logits,
                     out_path, niter):
    """Generate 4-column LRP comparison figure for the current NCP model state."""
    import glob as _glob
    import torchvision.transforms as _T
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt
    from PIL import Image as _Image

    tt = _T.Compose([
        _T.ToTensor(),
        _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rc = _T.Compose([_T.Resize(256), _T.CenterCrop(224)])

    col_titles = [
        "Image",
        "VGG16 ImageNet\nLRP (logit 430)",
        f"NCP pruned (ss0+1+2 ablated)\nLRP (basketball logit)  iter={niter}",
        "Vanilla pruned\nLRP (basketball logit)",
    ]

    fig, axes = _plt.subplots(len(img_paths), 4, figsize=(12, 3.2 * len(img_paths)))
    if len(img_paths) == 1:
        axes = [axes]

    for row, (p, pil_img, hm_im, logit_im, hm_pr, logit_pr) in enumerate(
            zip(img_paths, pil_images, van_im_hms, van_im_logits, van_pr_hms, van_pr_logits)):
        img_orig = _Image.open(p).convert('RGB')
        x = tt(rc(img_orig)).to(device)
        hm_ncp = _lrp_pruned_ncp(ncp_model, x, 0).sum(axis=0)
        logit_ncp = float(ncp_model(x.unsqueeze(0)).squeeze()[0])

        title_row = col_titles if row == 0 else [None]*4
        _show_image(axes[row][0], img_orig, ylabel=f"img-{row}", title=title_row[0])
        _show_heatmap(axes[row][1], hm_im,  logit=logit_im,  title=title_row[1])
        _show_heatmap(axes[row][2], hm_ncp, logit=logit_ncp, title=title_row[2])
        _show_heatmap(axes[row][3], hm_pr,  logit=logit_pr,  title=title_row[3])

    fig.suptitle(
        f"LRP-α1β0: VGG16 ImageNet vs NCP (ss0+1+2 ablated, iter {niter}) vs Vanilla pruned",
        fontsize=10,
    )
    _plt.tight_layout()
    _plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    _plt.close(fig)
    print(f"[heatmap] saved iter {niter} → {out_path}")


# ---------------------------------------------------------------------------
# Heatmap-generating subclass
# ---------------------------------------------------------------------------

class BasketballPruningFineTuner(PruningFineTuner):
    """Overrides get_candidates_to_prune to use a basketball-only rank loader.

    Fine-tuning after each prune step continues to use the full (both-class)
    train_loader so the model recovers on all examples.
    """

    def set_rank_loader(self, loader: DataLoader):
        self._rank_loader = loader

    def get_candidates_to_prune(self, num_filters_to_prune: int) -> list:
        if not hasattr(self, '_rank_loader'):
            raise RuntimeError("Call set_rank_loader() before pruning.")
        # Temporarily swap: ranking uses basketball-only loader
        _saved_loader = self.train_loader
        _saved_num = self.train_num
        # Also disable the rank_n sample cap so it doesn't interfere with a
        # rank_loader that is already bounded to exactly rank_n_images samples.
        _saved_rank_n = getattr(self.args, 'rank_n', None)
        self.args.rank_n = None
        self.train_loader = self._rank_loader
        self.train_num = len(self._rank_loader)
        try:
            result = super().get_candidates_to_prune(num_filters_to_prune)
        finally:
            self.train_loader = _saved_loader
            self.train_num = _saved_num
            self.args.rank_n = _saved_rank_n
        return result


class HeatmapBasketballPruningFineTuner(BasketballPruningFineTuner):
    """Generates a per-iteration LRP comparison heatmap after every test() call."""

    def set_heatmap_params(self, img_paths, pil_images,
                           van_im_hms, van_im_logits,
                           van_pr_hms, van_pr_logits,
                           heatmap_out_dir, device):
        self._hm_img_paths     = img_paths
        self._hm_pil_images    = pil_images
        self._hm_van_im_hms    = van_im_hms
        self._hm_van_im_logits = van_im_logits
        self._hm_van_pr_hms    = van_pr_hms
        self._hm_van_pr_logits = van_pr_logits
        self._hm_out_dir       = Path(heatmap_out_dir)
        self._hm_device        = device
        self._hm_out_dir.mkdir(parents=True, exist_ok=True)

    def test(self, compute_subgroup=False):
        super().test(compute_subgroup=compute_subgroup)
        if not hasattr(self, '_hm_img_paths'):
            return
        out_path = self._hm_out_dir / f'lrp_iter{self.niter:03d}.png'
        generate_heatmap(
            ncp_model=self.model,
            device=self._hm_device,
            img_paths=self._hm_img_paths,
            pil_images=self._hm_pil_images,
            van_im_hms=self._hm_van_im_hms,
            van_im_logits=self._hm_van_im_logits,
            van_pr_hms=self._hm_van_pr_hms,
            van_pr_logits=self._hm_van_pr_logits,
            out_path=out_path,
            niter=self.niter,
        )


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description='Basketball NCP pruning with positive-class LRP ranking.')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')

    # pruning config
    parser.add_argument('--relevance', action='store_true', default=True)
    parser.add_argument('--no_relevance', dest='relevance', action='store_false')
    parser.add_argument('--method_type', type=str, default='lrp')
    parser.add_argument('--pr_step', type=float, default=0.05)
    parser.add_argument('--total_pr', type=float, default=0.80)
    parser.add_argument('--rank_n', type=int, default=500,
                        help='Max samples used for ranking per iteration. '
                             'Set to 500 to match the basketball rank loader size.')
    parser.add_argument('--rank_n_images', type=int, default=500,
                        help='Number of basketball-class images to use for LRP ranking.')

    # warmup config
    parser.add_argument('--warmup_epochs', type=int, default=15,
                        help='Max epochs for warmup head training.')
    parser.add_argument('--warmup_patience', type=int, default=3,
                        help='Early-stop patience (epochs without val-loss improvement).')
    parser.add_argument('--warmup_min_delta', type=float, default=1e-3,
                        help='Minimum val-loss improvement to reset patience counter.')

    # model / experiment config
    parser.add_argument('--pruner', type=str, choices=['ncp', 'vanilla'], default='ncp')
    parser.add_argument('--data_type', type=str, default='basketball_imagenet')
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--fine_tune_conv_layers', action='store_true', default=True)
    parser.add_argument('--no_fine_tune_conv_layers', dest='fine_tune_conv_layers',
                        action='store_false')
    parser.add_argument('--fine_tune_without_augmented_layers', action='store_true',
                        default=True)
    parser.add_argument('--fine_tune_with_augmented_layers',
                        dest='fine_tune_without_augmented_layers', action='store_false')
    parser.add_argument('--subspace_dims', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--irrelevant_subspaces', type=int, nargs='+', default=[])
    parser.add_argument('--u_filepath', type=str,
                        default='/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt')
    parser.add_argument('--basketball_root', type=str,
                        default='/n/fs/ncp/NCP.v2/data/images/imagenet_430_binary',
                        help='Root dir for the basketball ImageFolder dataset.')
    parser.add_argument('--out_dir', type=str,
                        default='/n/fs/ncp/NCP.v2/results/basketball-ncp/')
    parser.add_argument('--final_finetune_epochs', type=int, default=5)
    parser.add_argument('--iter_finetune_epochs', type=int, default=2,
                        help='Fine-tuning epochs after each pruning iteration.')
    parser.add_argument('--stats_only', action='store_true', default=False)

    # Per-iteration LRP heatmap generation
    parser.add_argument('--iter_heatmap_dir', type=str, default=None,
                        help='If set, generate an LRP comparison heatmap after every '
                             'test() call and save to this directory. '
                             'Vanilla-imagenet and vanilla-pruned columns are cached once.')
    parser.add_argument('--vanilla_pruned_path', type=str, default=None,
                        help='Path to vanilla-pruned .pth checkpoint (for heatmap col 3). '
                             'Required when --iter_heatmap_dir is set.')
    parser.add_argument('--heatmap_img_dir', type=str,
                        default='/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images',
                        help='Directory of test images to use for LRP heatmaps.')

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Warmup: train classifier head with early stopping on val loss
# ---------------------------------------------------------------------------

def warmup_head(model, train_loader, val_loader, args, criterion):
    """Train classifier[6] only, for up to warmup_epochs with patience-based early stop.

    Bypasses the augmented path during warmup (raw conv features → head) so the
    head learns from the pre-ablation representation, consistent with run_PFT.py.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[6].parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=args.lr)

    _was_augmented = getattr(model, 'augmented', False)
    if _was_augmented:
        model.augmented = False

    best_val_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(args.warmup_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            data, target = batch[0], batch[1]
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Warmup epoch {epoch+1} | batch {batch_idx} | loss {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation loss for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data, target = batch[0], batch[1]
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
        val_loss /= len(val_loader.dataset)
        model.train()

        print(f"Warmup epoch {epoch+1}/{args.warmup_epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

        if best_val_loss - val_loss > args.warmup_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.warmup_patience})")
            if patience_counter >= args.warmup_patience:
                print(f"  Early stop at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    if _was_augmented:
        model.augmented = True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('CUDA:', torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # ImageNet normalisation transform (shared across all loaders)
    # ------------------------------------------------------------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if args.pruner == 'ncp':
        U = _load_u_matrix(args.u_filepath)
        U_ab, U_ab_T = ablate_subspace_matrix(U, args.subspace_dims, args.irrelevant_subspaces)
        model = AugmentedVGG16(U_ab, U_ab_T)
    else:
        model = models.vgg16(pretrained=(args.pretrained_model_path is None))

    model.classifier[6] = nn.Linear(4096, 2)

    if args.pretrained_model_path:
        print(f"Loading pretrained weights from: {args.pretrained_model_path}")
        sd = _load_checkpoint_state_dict(args.pretrained_model_path)
        if args.pruner == 'ncp':
            sd = _remap_vgg16_features_to_augmented(sd)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            for param in model.encode.parameters():
                param.requires_grad = False
            for param in model.decode.parameters():
                param.requires_grad = False
            if missing:
                print(f"  Missing keys (expected encode/decode only): {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
        else:
            model.load_state_dict(sd, strict=True)
        print("  Pretrained weights loaded.")

    if args.cuda:
        model = model.cuda()

    # ------------------------------------------------------------------
    # Build PruningFineTuner (sets up train_loader / test_loader from
    # get_basketball_imagenet via the basketball_imagenet data_type branch)
    # ------------------------------------------------------------------
    if args.pruner == 'ncp':
        adapter = AugmentedVGGAdapter(model)
    else:
        adapter = VanillaVGGAdapter(model)

    # ------------------------------------------------------------------
    # Per-iteration heatmap setup (if requested)
    # ------------------------------------------------------------------
    _heatmap_ready = False
    if args.iter_heatmap_dir:
        if not args.vanilla_pruned_path:
            raise ValueError("--vanilla_pruned_path is required when --iter_heatmap_dir is set")
        device_str = 'cuda' if args.cuda else 'cpu'

        print("[heatmap] loading vanilla ImageNet VGG16")
        vanilla_imagenet = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).to(device_str).eval()
        for p in vanilla_imagenet.parameters():
            p.requires_grad_(False)

        print(f"[heatmap] loading vanilla pruned model from {args.vanilla_pruned_path}")
        van_ckpt = torch.load(args.vanilla_pruned_path, map_location='cpu', weights_only=False)
        vanilla_pruned = van_ckpt['model'].to(device_str).eval()
        for p in vanilla_pruned.parameters():
            p.requires_grad_(False)

        img_paths = sorted(
            glob.glob(str(Path(args.heatmap_img_dir) / '*.jpg'))
            + glob.glob(str(Path(args.heatmap_img_dir) / '*.png'))
            + glob.glob(str(Path(args.heatmap_img_dir) / '*.JPEG'))
        )
        assert len(img_paths) > 0, f"No images found in {args.heatmap_img_dir}"
        print(f"[heatmap] found {len(img_paths)} test images")

        pil_images, van_im_hms, van_im_logits, van_pr_hms, van_pr_logits = \
            precompute_static_heatmaps(img_paths, vanilla_imagenet, vanilla_pruned, device_str)

        tuner_cls = HeatmapBasketballPruningFineTuner
        _heatmap_ready = True
    else:
        tuner_cls = BasketballPruningFineTuner

    tuner = tuner_cls(args, model, adapter)

    if _heatmap_ready:
        tuner.set_heatmap_params(
            img_paths, pil_images,
            van_im_hms, van_im_logits,
            van_pr_hms, van_pr_logits,
            args.iter_heatmap_dir, device_str,
        )

    # ------------------------------------------------------------------
    # Build rank_loader: first 500 basketball-class images by filename
    # ------------------------------------------------------------------
    rank_loader = build_basketball_rank_loader(
        root_dir=args.basketball_root,
        transform=imagenet_transform,
        n=args.rank_n_images,
        batch_size=args.train_batch_size,
        num_workers=3 if args.cuda else 0,
        pin_memory=args.cuda,
    )
    tuner.set_rank_loader(rank_loader)

    # Persist eval supplement log (always empty for basketball, but kept for
    # consistency with run_PFT.py so downstream tooling can read the same file)
    os.makedirs(args.out_dir, exist_ok=True)
    supplement_log_path = os.path.join(args.out_dir, 'eval_supplement_fnames.txt')
    with open(supplement_log_path, 'w') as _f:
        _f.write("# Filenames pulled from official TEST split to supplement eval set\n")
        _f.write(f"# min_male_with_attr=0  count=0\n")
        _f.write("# (not applicable for basketball_imagenet)\n")

    # ------------------------------------------------------------------
    # Warmup: train 2-d head on full training set (both classes)
    # ------------------------------------------------------------------
    print("=== Warmup: training classifier head ===")
    criterion = nn.CrossEntropyLoss()
    warmup_head(model, tuner.train_loader, tuner.test_loader, args, criterion)

    # Re-enable all parameters for the pruning phase (adapter will set them up)
    for param in model.parameters():
        param.requires_grad = True

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------
    print("=== Pruning ===")
    tuner.prune(
        fine_tune_conv_layers=args.fine_tune_conv_layers,
        fine_tune_without_augmented_layers=args.fine_tune_without_augmented_layers,
        final_finetune_epochs=args.final_finetune_epochs,
        iter_finetune_epochs=args.iter_finetune_epochs,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pruned_structure = [
        m.out_channels
        for layer_idx, m in enumerate(adapter.iter_modules_forward_order())
        if isinstance(m, nn.Conv2d) and adapter.is_prunable_conv(m, layer_idx)
    ]

    out_path = os.path.join(
        args.out_dir,
        'ncp-2.pth' if args.pruner == 'ncp' else 'van.pth',
    )
    save_dict = {
        'pruned_structure': pruned_structure,
        'train_loss': tuner.train_loss_tot,
        'train_acc': tuner.train_acc_tot,
        'test_loss': tuner.test_loss_tot,
        'test_acc': tuner.test_acc_tot,
        'test_iter': tuner.test_iter,
        'test_precision_per_class': tuner.test_precision_tot,
        'test_recall_per_class': tuner.test_recall_tot,
        'subgroup_stats': tuner.subgroup_stats_tot,
        'irrelevant_subspaces': args.irrelevant_subspaces,
        'rank_n_images': args.rank_n_images,
    }
    if not args.stats_only:
        save_dict['model'] = tuner.model
        save_dict['state_dict'] = tuner.model.state_dict()
    torch.save(save_dict, out_path)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
