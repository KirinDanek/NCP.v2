#!/usr/bin/env python3
"""
visualize_lips_subspace.py

For each sx config, generate 4 montage figures (one per subgroup):
  female + lipstick  |  female + no lipstick
  male   + lipstick  |  male   + no lipstick

Each figure shows the top-N examples sorted by lips-subspace relevance (highest first).
Each row: original image | full LRP heatmap (pixel-level) | lips subspace heatmap (28x28→224).

LRP method: alphabeta (α=1, β=0), with encode/decode using gamma(0.0) per AugmentedVGGAdapter.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch import nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AugmentedVGG16 import AugmentedVGG16
from run_PFT import _load_checkpoint_state_dict, _remap_vgg16_features_to_augmented, _load_u_matrix
from data import get_celeba_attribute_splits_for_pruning
from lrp import lrp as lrp_fn
from prune_vgg import AugmentedVGGAdapter

PRETRAINED = (
    "/n/fs/ncp/drsa-demo/fairness_harness_outputs/"
    "Wearing_Lipstick/seed0/checkpoints/best_model.pth"
)
U_BASE = (
    "/n/fs/ncp/drsa-demo/data/projection_matrices/"
    "celeba/Wearing_Lipstick/seed0/conv4_3"
)
OUT_DIR = "/n/fs/ncp/NCP.v2/results"

# Use sx4 by default; change to run others
CONFIGS = [
    {
        "name": "sx4",
        "u_path": os.path.join(U_BASE, "U_sx4.npy"),
        "subspace_dims": [128, 128, 128, 128],
        "irrelevant_subspaces": [0, 1, 3],
    },
    {
        "name": "sx8",
        "u_path": os.path.join(U_BASE, "U_sx8.npy"),
        "subspace_dims": [64, 64, 64, 64, 64, 64, 64, 64],
        "irrelevant_subspaces": [0, 1, 2, 3, 5, 6, 7],
    },
    {
        "name": "sx16",
        "u_path": os.path.join(U_BASE, "U_sx16.npy"),
        "subspace_dims": [32] * 16,
        "irrelevant_subspaces": [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15],
    },
]

TOP_N    = 8   # examples per subgroup
BATCH_SIZE = 32

# Transform for model input (normalized)
NORM_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Transform for display (no normalize)
VIZ_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


# ── model / LRP helpers ───────────────────────────────────────────────────────

def make_relevant_mask(subspace_dims, irrelevant_subspaces):
    irrel_set = set(irrelevant_subspaces)
    mask = torch.zeros(sum(subspace_dims), dtype=torch.bool)
    start = 0
    for i, sz in enumerate(subspace_dims):
        if i not in irrel_set:
            mask[start : start + sz] = True
        start += sz
    return mask


def build_module_list(model):
    mods = list(model.before)
    if model.augmented:
        mods += [model.encode, model.decode]
    mods += list(model.after)
    mods += list(model.classifier)
    return mods


def build_lrp_rules(model, module_list):
    adapter = AugmentedVGGAdapter(model)
    total = len(module_list)
    rev = list(reversed(module_list))
    return [
        adapter.get_backward_lrp_rule(mod, i, total, 'alphabeta', 1.0)
        for i, mod in enumerate(rev)
    ]


def load_model(U, device):
    U = U.float()
    model = AugmentedVGG16(U=U, UT=U.t().contiguous())
    model.classifier[6] = nn.Linear(4096, 2)
    model = model.to(device)
    model.augmented = True
    sd = _load_checkpoint_state_dict(PRETRAINED)
    sd = _remap_vgg16_features_to_augmented(sd)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def register_hooks(module_list):
    handles = []
    for m in module_list:
        def _hook(mod, inp, out):
            mod.input = inp[0]
            mod.output = out.data
        handles.append(m.register_forward_hook(_hook))
    return handles


# ── pass 1: collect scores for all examples ───────────────────────────────────

def collect_scores(model, loader, rel_mask, device, lrp_rules):
    """Return (scores, labels, genders) numpy arrays over the full dataset."""
    module_list  = build_module_list(model)
    decode_idx   = module_list.index(model.decode)
    rev_mods     = list(reversed(module_list))

    all_scores, all_labels, all_genders = [], [], []

    for batch_idx, (x, y, g) in enumerate(loader):
        x = x.to(device)
        handles = register_hooks(module_list)
        with torch.no_grad():
            logits = model(x)
        for h in handles: h.remove()

        R = torch.zeros_like(logits)
        R[:, 1] = logits[:, 1]

        encode_R = None
        for mi, (mod, (lv, lp)) in enumerate(zip(rev_mods, lrp_rules)):
            R = lrp_fn(mod, R, lrp_var=lv, param=lp)
            if (len(module_list) - 1 - mi) == decode_idx:
                encode_R = R.clone()

        rm = rel_mask.to(encode_R.device)
        if encode_R.dim() == 4:
            scores = encode_R[:, rm, :, :].sum(dim=(1, 2, 3))
        else:
            scores = encode_R[:, rm].sum(dim=1)

        all_scores.append(scores.cpu().float().numpy())
        all_labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
        all_genders.append(g.numpy() if isinstance(g, torch.Tensor) else np.array(g))

        if (batch_idx + 1) % 100 == 0:
            print(f"  scoring batch {batch_idx+1}/{len(loader)}", flush=True)

    return (np.concatenate(all_scores),
            np.concatenate(all_labels),
            np.concatenate(all_genders))


# ── pass 2: per-example LRP for spatial heatmaps ─────────────────────────────

def lrp_single(model, x_single, rel_mask, device, lrp_rules):
    """
    Run forward + alphabeta LRP for one example (x_single: 1×3×224×224 tensor).
    Returns:
      lips_hm : (224, 224) numpy — lips subspace relevance at 28×28 upsampled
      std_hm  : (224, 224) numpy — full pixel-level LRP (sum over channels)
    """
    module_list = build_module_list(model)
    decode_idx  = module_list.index(model.decode)
    rev_mods    = list(reversed(module_list))

    handles = register_hooks(module_list)
    with torch.no_grad():
        logits = model(x_single.to(device))
    for h in handles: h.remove()

    R = torch.zeros_like(logits)
    R[:, 1] = logits[:, 1]

    encode_R = None
    for mi, (mod, (lv, lp)) in enumerate(zip(rev_mods, lrp_rules)):
        R = lrp_fn(mod, R, lrp_var=lv, param=lp)
        if (len(module_list) - 1 - mi) == decode_idx:
            encode_R = R.clone()

    # Lips subspace: sum relevant channels → 28×28 → upsample to 224×224
    rm = rel_mask.to(encode_R.device)
    lips_28 = encode_R[:, rm, :, :].sum(dim=1, keepdim=True)    # (1,1,28,28)
    lips_hm = F.interpolate(lips_28, size=(224, 224), mode='bilinear',
                            align_corners=False)[0, 0].cpu().numpy()

    # Full pixel-level LRP: R is now (1,3,224,224) — sum over colour channels
    std_hm = R.sum(dim=1)[0].cpu().numpy()   # (224, 224)

    return lips_hm, std_hm, logits[0, 1].item()


# ── plotting helpers ──────────────────────────────────────────────────────────

def _heatmap_ax(ax, hm, title="", ref_hm=None):
    """Seismic heatmap matching cxai viz.heatmap style."""
    if ref_hm is None:
        ref_hm = hm
    b = np.abs(ref_hm).max() + 1e-9
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, :3] *= 0.85
    cmap = mcolors.ListedColormap(my_cmap)
    ax.imshow(hm, cmap=cmap, vmin=-b, vmax=b)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def make_montage(rows_data, subgroup_name, sx_name, out_path, top_n):
    """
    rows_data: list of (pil_img, lips_hm, std_hm, score, logit) for top_n examples.
    """
    ncols = 3
    nrows = len(rows_data)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Full LRP", f"Lips subspace\n({sx_name})"]

    for row_i, (pil_img, lips_hm, std_hm, score, logit) in enumerate(rows_data):
        # Column 0: original image
        axes[row_i, 0].imshow(np.array(pil_img))
        axes[row_i, 0].set_xticks([]); axes[row_i, 0].set_yticks([])
        axes[row_i, 0].set_ylabel(f"score={score:.3f}\nlogit={logit:.2f}", fontsize=6)

        # Column 1: standard LRP heatmap
        _heatmap_ax(axes[row_i, 1], std_hm)

        # Column 2: lips subspace heatmap (normalise relative to itself)
        _heatmap_ax(axes[row_i, 2], lips_hm)

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=8, fontweight='bold')

    fig.suptitle(f"{sx_name} — {subgroup_name} (top {top_n} by lips relevance)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

SUBGROUPS = [
    ("female + lipstick",    0, 1),
    ("female + no lipstick", 0, 0),
    ("male   + lipstick",    1, 1),
    ("male   + no lipstick", 1, 0),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading prune_val split...")
    _, prune_val_ds, _ = get_celeba_attribute_splits_for_pruning(min_male_with_attr=200)
    # Access underlying CelebAAttributeDataset to get image paths
    base_ds = prune_val_ds.base

    loader = torch.utils.data.DataLoader(
        prune_val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    print(f"  prune_val size: {len(prune_val_ds)}\n")

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"=== {name} ===")

        U_tensor = _load_u_matrix(cfg["u_path"])
        model    = load_model(U_tensor, device)
        rel_mask = make_relevant_mask(cfg["subspace_dims"], cfg["irrelevant_subspaces"])
        mod_list = build_module_list(model)
        rules    = build_lrp_rules(model, mod_list)

        # Pass 1: score all examples
        print("  Pass 1: scoring all examples...")
        scores, labels, genders = collect_scores(model, loader, rel_mask, device, rules)

        # For each subgroup, find top-N indices by lips relevance
        for (sg_name, sg_gender, sg_label) in SUBGROUPS:
            mask = (genders == sg_gender) & (labels == sg_label)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                print(f"  [{sg_name}] no examples, skipping")
                continue

            # Sort by score descending, take top_n
            sorted_idxs = idxs[np.argsort(scores[idxs])[::-1]]
            top_idxs    = sorted_idxs[:TOP_N]

            print(f"  [{sg_name}] n={len(idxs)}, "
                  f"score range [{scores[idxs].min():.3f}, {scores[idxs].max():.3f}]")

            # Pass 2: per-example LRP for top-N
            rows_data = []
            for idx in top_idxs:
                fname = base_ds.fnames[idx]
                img_path = base_ds.image_path(fname)

                pil_img = Image.open(img_path).convert("RGB")
                pil_display = VIZ_TRANSFORM(pil_img)      # PIL, cropped, no normalize
                x_norm = NORM_TRANSFORM(pil_img).unsqueeze(0)  # (1,3,224,224)

                lips_hm, std_hm, logit = lrp_single(model, x_norm, rel_mask, device, rules)
                rows_data.append((pil_display, lips_hm, std_hm, scores[idx], logit))

            safe_name = sg_name.strip().replace(" ", "_").replace("+", "plus")
            out_path = os.path.join(OUT_DIR, f"lips_montage_{name}_{safe_name}.png")
            make_montage(rows_data, sg_name, name, out_path, len(rows_data))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
