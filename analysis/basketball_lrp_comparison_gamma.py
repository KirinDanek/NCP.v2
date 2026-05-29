"""
basketball_lrp_comparison_gamma.py

Single comparison figure: 4 test images × 4 columns
  col 0 — original image
  col 1 — vanilla ImageNet VGG16 LRP heatmap      (w.r.t. logit 430, ImageNet basketball)
  col 2 — NCP pruned VGG16 LRP heatmap            (w.r.t. logit 0,   binary basketball class)
                                                    using irrelevant_subspaces=[3]
  col 3 — Vanilla LRP pruned VGG16 LRP heatmap    (w.r.t. logit 0,   binary basketball class)

All use block-dependent LRP-gamma throughout, with the 'first' rule at the pixel layer:
  block 6 (classifier FC)  : gamma=0.01
  block 5 (conv5_*)        : gamma=0.01
  block 4 (conv4_*)        : gamma=0.10
  block 3 (conv3_*)        : gamma=0.25
  blocks 1-2 (conv1/2_*)   : gamma=0.50
  pixel layer (conv1_1)    : 'first' (z_B rule)

Blocks are determined by MaxPool2d boundaries in the features list.
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

sys.path.insert(0, "/n/fs/ncp/NCP.v2")

from lrp import lrp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULT_DIR        = Path("/n/fs/ncp/NCP.v2/results/basketball-ncp")
VANILLA_RESULT_DIR = Path("/n/fs/ncp/NCP.v2/results/basketball-vanilla")
IMG_DIR           = Path("/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images")
OUT_PATH          = RESULT_DIR / "lrp_comparison_subspace3_gamma.png"

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
# LRP-gamma with block-dependent gamma — shared logic
# ---------------------------------------------------------------------------

# Block index (1–5 = conv blocks, 6 = classifier) → gamma value
GAMMA_BY_BLOCK = {1: 0.5, 2: 0.5, 3: 0.25, 4: 0.1, 5: 0.01, 6: 0.01}


def _build_gamma_map(feature_modules, classifier_modules, first_conv):
    """Return {id(module): (lrp_var, param)} for every module.

    Block boundaries are detected by MaxPool2d layers in feature_modules,
    exactly mirroring the VGG16 architecture (pool after blocks 1–5).
    For AugmentedVGG16, pass list(model.before) + list(model.after) as
    feature_modules — concatenation reconstructs the full features sequence.
    """
    m2rule = {}
    block = 1
    for m in feature_modules:
        if m is first_conv:
            m2rule[id(m)] = ('first', None)
        else:
            m2rule[id(m)] = ('gamma', GAMMA_BY_BLOCK[block])
        if isinstance(m, nn.MaxPool2d):
            block += 1
    for m in classifier_modules:
        m2rule[id(m)] = ('gamma', GAMMA_BY_BLOCK[6])
    return m2rule


def _run_lrp_gamma(modules_bwd, m2rule, R):
    """Walk modules in backward order using per-block gamma (or 'first')."""
    for m in modules_bwd:
        lrp_var, param = m2rule[id(m)]
        R = lrp(m, R, lrp_var=lrp_var, param=param)
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

    first_conv = feature_layers[0]   # features[0] = conv1_1
    m2rule     = _build_gamma_map(feature_layers, classifier_layers, first_conv)
    modules_bwd = list(reversed(classifier_layers)) + list(reversed(feature_layers))

    R = _run_lrp_gamma(modules_bwd, m2rule, R)

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

    first_conv  = feature_layers[0]
    m2rule      = _build_gamma_map(feature_layers, classifier_layers, first_conv)
    modules_bwd = list(reversed(classifier_layers)) + list(reversed(feature_layers))

    R = _run_lrp_gamma(modules_bwd, m2rule, R)

    for h in handles:
        h.remove()
    return R.squeeze(0).cpu().numpy()   # (3, H, W)


def lrp_pruned_ncp(model, x_tensor, target_class):
    """LRP heatmap for a pruned AugmentedVGG16 (binary classifier).

    before + after reconstructs the full VGG16 features sequence, so
    _build_gamma_map sees the same MaxPool2d boundaries as vanilla VGG16.
    """
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

    feature_layers = before_layers + after_layers
    first_conv     = next(m for m in before_layers if isinstance(m, nn.Conv2d))
    m2rule         = _build_gamma_map(feature_layers, classifier_layers, first_conv)
    modules_bwd    = (list(reversed(classifier_layers))
                      + list(reversed(after_layers))
                      + list(reversed(before_layers)))

    R = _run_lrp_gamma(modules_bwd, m2rule, R)

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
        "LRP-γ heatmaps (γ: 0.5/0.5/0.25/0.1/0.01/0.01): VGG16 ImageNet vs NCP (subspace 3) vs Vanilla pruned",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
