"""
basketball_lrp_comparison_simple.py

Single comparison figure: 4 test images × 4 columns
  col 0 — original image
  col 1 — vanilla ImageNet VGG16 LRP heatmap      (w.r.t. logit 430, ImageNet basketball)
  col 2 — NCP pruned VGG16 LRP heatmap            (w.r.t. logit 0,   binary basketball class)
                                                    using irrelevant_subspaces=[3]
  col 3 — Vanilla LRP pruned VGG16 LRP heatmap    (w.r.t. logit 0,   binary basketball class)

All use LRP-simple (epsilon≈0) throughout all layers, including the pixel layer.
avgpool in vanilla VGG16 is identity for 224px inputs and is skipped in the
backward LRP chain; the reshape from flat (1, 25088) to (1, 512, 7, 7) is
handled automatically by lrp.Pooling reading pool5.output.shape.
"""

import sys
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from lrp import lrp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULT_DIR        = Path("/n/fs/ncp/NCP.v2/results/basketball-ncp")
VANILLA_RESULT_DIR = Path("/n/fs/ncp/NCP.v2/results/basketball-vanilla")
IMG_DIR           = Path("/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images")
OUT_PATH          = RESULT_DIR / "lrp_comparison_subspace3_simple.png"

IMAGENET_BBALL    = 430   # ImageNet class index for basketball
NCP_BBALL         = 0     # binary class index in the pruned model
VAN_BBALL         = 0     # binary class index in the vanilla pruned model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

input_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
rc_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])


# ---------------------------------------------------------------------------
# Forward hook
# ---------------------------------------------------------------------------

def fhook(module, input, output):
    module.input  = input[0]
    module.output = output.data


# ---------------------------------------------------------------------------
# LRP (alpha-1 beta-0) — shared logic
# ---------------------------------------------------------------------------

def _run_lrp(modules_bwd, first_conv, R):
    """Walk modules in backward order, applying LRP-simple (epsilon≈0), with 'first' at pixel layer."""
    for m in modules_bwd:
        if m is first_conv:
            R = lrp(m, R, lrp_var='first', param=None)
        else:
            R = lrp(m, R, lrp_var='simple', param=None)
    return R


def lrp_vanilla_vgg16(model, x_tensor, target_class):
    """LRP heatmap for a standard torchvision VGG16 (1000-class).

    avgpool is identity for 224px inputs and is excluded from the LRP chain.
    """
    # Register hooks on features + classifier (skip avgpool — it's identity
    # for 224px and AdaptiveAvgPool2d is not handled by lrp.py)
    feature_layers     = list(model.features)
    classifier_layers  = list(model.classifier)
    all_hooked = feature_layers + classifier_layers

    handles = [m.register_forward_hook(fhook) for m in all_hooked]
    model.eval()
    with torch.no_grad():
        out = model(x_tensor.unsqueeze(0))   # (1, 1000)

    R = torch.zeros_like(out)
    R[0, target_class] = 1.0

    # Backward: classifier reversed, then features reversed
    modules_bwd = list(reversed(classifier_layers)) + list(reversed(feature_layers))
    first_conv  = feature_layers[0]   # features[0] = conv1_1

    R = _run_lrp(modules_bwd, first_conv, R)

    for h in handles:
        h.remove()
    return R.squeeze(0).cpu().numpy()   # (3, H, W)


def lrp_pruned_vanilla(model, x_tensor, target_class):
    """LRP heatmap for a vanilla-pruned VGG16 (binary classifier).

    The model retains the standard torchvision features + classifier structure,
    so the same walk as lrp_vanilla_vgg16 applies, but with target_class=0
    (binary basketball class) and 2 output logits.
    """
    feature_layers    = list(model.features)
    classifier_layers = list(model.classifier)
    all_hooked = feature_layers + classifier_layers

    handles = [m.register_forward_hook(fhook) for m in all_hooked]
    model.eval()
    with torch.no_grad():
        out = model(x_tensor.unsqueeze(0))   # (1, 2)

    R = torch.zeros_like(out)
    R[0, target_class] = 1.0

    modules_bwd = list(reversed(classifier_layers)) + list(reversed(feature_layers))
    first_conv  = feature_layers[0]

    R = _run_lrp(modules_bwd, first_conv, R)

    for h in handles:
        h.remove()
    return R.squeeze(0).cpu().numpy()   # (3, H, W)


def lrp_pruned_ncp(model, x_tensor, target_class):
    """LRP heatmap for a pruned AugmentedVGG16 (binary classifier)."""
    before_layers     = list(model.before)
    after_layers      = list(model.after)
    classifier_layers = list(model.classifier)
    all_hooked = before_layers + after_layers + classifier_layers

    handles = [m.register_forward_hook(fhook) for m in all_hooked]
    model.eval()
    with torch.no_grad():
        out = model(x_tensor.unsqueeze(0))   # (1, 2)

    R = torch.zeros_like(out)
    R[0, target_class] = 1.0

    modules_bwd = (list(reversed(classifier_layers))
                   + list(reversed(after_layers))
                   + list(reversed(before_layers)))
    first_conv  = next(m for m in before_layers if isinstance(m, nn.Conv2d))

    R = _run_lrp(modules_bwd, first_conv, R)

    for h in handles:
        h.remove()
    return R.squeeze(0).cpu().numpy()   # (3, H, W)


# ---------------------------------------------------------------------------
# Viz
# ---------------------------------------------------------------------------

def show_image(ax, pil_img, ylabel=None, title=None):
    ax.imshow(rc_transform(pil_img))
    ax.set_xticks([]); ax.set_yticks([])
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title:  ax.set_title(title,   fontsize=9)


def show_heatmap(ax, heatmap_2d, logit=None, title=None):
    b = np.abs(heatmap_2d).max() or 1.0
    cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    cmap[:, :3] *= 0.85
    ax.imshow(heatmap_2d, cmap=ListedColormap(cmap), vmin=-b, vmax=b)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)
    xlabel = f"$\\sum R_i$={heatmap_2d.sum():.3f}"
    if logit is not None:
        xlabel += f"  logit={logit:.3f}"
    ax.set_xlabel(xlabel, fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[device] {DEVICE}")

    img_paths = sorted(
        glob.glob(str(IMG_DIR / "*.jpg"))
        + glob.glob(str(IMG_DIR / "*.png"))
        + glob.glob(str(IMG_DIR / "*.JPEG"))
    )
    assert len(img_paths) == 4, f"Expected 4 test images, found {len(img_paths)}"

    # Load vanilla VGG16
    print("[loading] vanilla ImageNet VGG16")
    vanilla = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
    ).to(DEVICE).eval()
    for p in vanilla.parameters():
        p.requires_grad_(False)

    # Load NCP pruned model (irrelevant_subspace=3)
    ckpt_path = RESULT_DIR / "irrelevant_subspace_3" / "ncp-2.pth"
    print(f"[loading] NCP model from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ncp = ckpt["model"].to(DEVICE).eval()

    # Load vanilla LRP pruned model
    van_ckpt_path = VANILLA_RESULT_DIR / "van.pth"
    print(f"[loading] vanilla pruned model from {van_ckpt_path}")
    van_ckpt = torch.load(str(van_ckpt_path), map_location="cpu")
    van_pruned = van_ckpt["model"].to(DEVICE).eval()
    for p in van_pruned.parameters():
        p.requires_grad_(False)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(12, 13))

    col_titles = [
        "Image",
        f"VGG16 ImageNet\nLRP (logit {IMAGENET_BBALL})",
        f"NCP pruned (subspace 3)\nLRP (basketball logit)",
        f"Vanilla pruned\nLRP (basketball logit)",
    ]

    for row, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert("RGB")
        x   = input_transform(img).to(DEVICE)

        # Vanilla ImageNet heatmap
        hm_van = lrp_vanilla_vgg16(vanilla, x, IMAGENET_BBALL).sum(axis=0)
        logit_van = float(vanilla(x.unsqueeze(0)).squeeze()[IMAGENET_BBALL])

        # NCP heatmap
        hm_ncp = lrp_pruned_ncp(ncp, x, NCP_BBALL).sum(axis=0)
        logit_ncp = float(ncp(x.unsqueeze(0)).squeeze()[NCP_BBALL])

        # Vanilla pruned heatmap
        hm_van_pruned = lrp_pruned_vanilla(van_pruned, x, VAN_BBALL).sum(axis=0)
        logit_van_pruned = float(van_pruned(x.unsqueeze(0)).squeeze()[VAN_BBALL])

        print(f"img-{row}: imagenet logit={logit_van:.3f}  ncp logit={logit_ncp:.3f}  "
              f"van_pruned logit={logit_van_pruned:.3f}  "
              f"imagenet_sum={hm_van.sum():.3f}  ncp_sum={hm_ncp.sum():.3f}  "
              f"van_pruned_sum={hm_van_pruned.sum():.3f}")

        title_row = col_titles if row == 0 else [None, None, None, None]

        show_image(axes[row, 0],   img,           ylabel=f"img-{row}", title=title_row[0])
        show_heatmap(axes[row, 1], hm_van,         logit=logit_van,         title=title_row[1])
        show_heatmap(axes[row, 2], hm_ncp,         logit=logit_ncp,         title=title_row[2])
        show_heatmap(axes[row, 3], hm_van_pruned,  logit=logit_van_pruned,  title=title_row[3])

    fig.suptitle(
        "LRP-simple heatmaps: VGG16 ImageNet vs NCP pruned (subspace 3) vs Vanilla pruned",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
