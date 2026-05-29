"""
basketball_lrp_heatmaps.py

Generates vanilla LRP heatmaps for 4 test images using the 3 pruned NCP models
(irrelevant_subspace_{1,2,3}).  Produces one PNG per model.

Layout per PNG: 4 rows x 2 columns — original image | LRP heatmap w.r.t. the
basketball logit (class index 0 in the pruned binary classifier).

Uses lrp.py directly on the AugmentedVGG16 model.  Rules:
  - before[0] (conv1_1, pixel layer): 'first'
  - all other layers:                 'alpha', param=1
  - ReLU / Dropout / MaxPool2d:       handled by lrp() internally
"""

import sys
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
RESULT_DIR       = Path("/n/fs/ncp/NCP.v2/results/basketball-ncp")
IMG_DIR          = Path("/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images")
OUT_DIR          = RESULT_DIR
BASKETBALL_LABEL = 0   # class 0 = basketball in the pruned binary model
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

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
# Forward hook (same as prune_vgg.py)
# ---------------------------------------------------------------------------

def fhook(module, input, output):
    module.input  = input[0]
    module.output = output.data


# ---------------------------------------------------------------------------
# LRP heatmap
# ---------------------------------------------------------------------------

def compute_lrp_heatmap(model, x_tensor, target_class):
    """Return pixel-level LRP relevance as a numpy array of shape (3, H, W).

    x_tensor: (3, H, W) float32 tensor on the correct device.

    Rules: LRP-alpha1-beta0 throughout.
      - before[0] (conv1_1, pixel layer) : 'first'
      - all other Conv2d and Linear      : 'alpha', param=1  (alpha=1, beta=0)
      - MaxPool2d / ReLU / Dropout       : handled by lrp() internally
    """
    modules_fwd = list(model.before) + list(model.after) + list(model.classifier)

    handles = [m.register_forward_hook(fhook) for m in modules_fwd]

    model.eval()
    with torch.no_grad():
        out = model(x_tensor.unsqueeze(0))   # (1, num_classes)

    R = torch.zeros_like(out)
    R[0, target_class] = 1.0

    first_conv = next(m for m in model.before if isinstance(m, nn.Conv2d))

    for m in reversed(modules_fwd):
        if m is first_conv:
            R = lrp(m, R, lrp_var='first', param=None)
        else:
            R = lrp(m, R, lrp_var='alpha', param=1)

    for h in handles:
        h.remove()

    return R.squeeze(0).cpu().numpy()   # (3, H, W)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def show_image(ax, pil_img):
    ax.imshow(rc_transform(pil_img))
    ax.set_xticks([])
    ax.set_yticks([])


def show_heatmap(ax, heatmap_2d, logit=None, title=""):
    """heatmap_2d: (H, W) numpy array."""
    b = np.abs(heatmap_2d).max()
    cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    cmap[:, :3] *= 0.85
    cmap = ListedColormap(cmap)

    ax.imshow(heatmap_2d, cmap=cmap, vmin=-b, vmax=b)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    label = f"$\\sum R_i$={heatmap_2d.sum():.3f}"
    if logit is not None:
        label += f"  logit={logit:.3f}"
    ax.set_xlabel(label, fontsize=7)


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
    print(f"[images] {[Path(p).name for p in img_paths]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for subspace_id in [1, 2, 3]:
        ckpt_path = RESULT_DIR / f"irrelevant_subspace_{subspace_id}" / "ncp-2.pth"
        print(f"\n[subspace {subspace_id}] loading {ckpt_path}")

        ckpt  = torch.load(str(ckpt_path), map_location="cpu")
        model = ckpt["model"].to(DEVICE).eval()

        # Confirm the pruned structure and head
        with torch.no_grad():
            dummy = input_transform(Image.open(img_paths[0]).convert("RGB"))
            logits_check = model(dummy.unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
        print(f"  sample logits (basketball=0, not_basketball=1): {logits_check}")

        fig, axes = plt.subplots(4, 2, figsize=(6, 13))

        for row, img_path in enumerate(img_paths):
            img = Image.open(img_path).convert("RGB")
            x   = input_transform(img).to(DEVICE)

            heatmap  = compute_lrp_heatmap(model, x, BASKETBALL_LABEL)  # (3, H, W)
            hmap_2d  = heatmap.sum(axis=0)                               # (H, W)

            with torch.no_grad():
                logits = model(x.unsqueeze(0)).squeeze().cpu().numpy()
            basketball_logit = float(logits[BASKETBALL_LABEL])
            print(f"  img-{row}: basketball logit={basketball_logit:.4f}  "
                  f"heatmap sum={hmap_2d.sum():.4f}")

            show_image(axes[row, 0], img)
            axes[row, 0].set_ylabel(f"img-{row}", fontsize=9)
            if row == 0:
                axes[row, 0].set_title("Image", fontsize=9)

            show_heatmap(axes[row, 1], hmap_2d, logit=basketball_logit,
                         title="LRP (basketball logit)" if row == 0 else "")

        fig.suptitle(
            f"NCP pruned VGG16 · irrelevant subspace {subspace_id} · "
            f"LRP w.r.t. basketball (class 0)",
            fontsize=10,
        )
        plt.tight_layout()

        out_path = OUT_DIR / f"lrp_heatmaps_irrelevant_subspace_{subspace_id}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
