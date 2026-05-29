"""
carton_crate_lrp_subspaces.py

Generates two DRSA subspace heatmap montages using vanilla ImageNet VGG16,
modelled on basketball_montage.py.

Figure 1 — carton (ImageNet logit 478):
  5 images from carton_dugong/prune_set/carton/n02971356_0000{0,1,3,4,5}.jpeg
  Projection matrix: U_carton_tensor.pt

Figure 2 — crate (ImageNet logit 519):
  4 images from crate_packet/imagenet_{WMC,NWMC}_NWMP/target/n03127925_*.JPEG
  Projection matrix: U_crate_tensor.pt

Layout per figure: N rows x (2 + K) columns
  col 0        — original image
  col 1        — standard LRP heatmap (sum over colour channels)
  col 2..(2+K) — per-subspace LRP heatmaps (canonical order 0..K-1)
"""

import sys
import collections
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/n/fs/ncp/NCP.v2")
sys.path.insert(0, "/n/fs/ncp/drsa-demo")

from cxai import factory, inspector, constants, models
from cxai import utils as putils

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CARTON_DIR = Path("/n/fs/ncp/NCP.v2/data/images/carton_dugong/prune_set/carton")
CRATE_BASE = Path("/n/fs/ncp/NCP.v2/data/images/crate_packet")
U_DIR      = Path("/n/fs/ncp/NCP.v2/data/projection_matrices")
OUT_DIR    = Path("/n/fs/ncp/NCP.v2/results")

LAYER   = "conv4_3"
K       = 4
USE_ABS = False
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

CARTON_IMGS = [
    CARTON_DIR / "n02971356_00000.jpeg",
    CARTON_DIR / "n02971356_00003.jpeg",
    CARTON_DIR / "n02971356_00004.jpeg",
    CARTON_DIR / "n02971356_00005.jpeg",
]
CARTON_TARGET = 478   # ImageNet "carton"
CARTON_U      = U_DIR / "U_carton_tensor.pt"
CARTON_OUT    = OUT_DIR / "carton_lrp_subspaces.png"

CRATE_IMGS = [
    CRATE_BASE / "imagenet_WMC_NWMP/target/n03127925_97.JPEG",
    CRATE_BASE / "imagenet_WMC_NWMP/target/n03127925_893.JPEG",
    CRATE_BASE / "imagenet_WMC_NWMP/target/n03127925_895.JPEG",
    CRATE_BASE / "imagenet_WMC_NWMP/target/n03127925_923.JPEG",
]
CRATE_TARGET = 519    # ImageNet "crate"
CRATE_U      = U_DIR / "U_crate_tensor.pt"
CRATE_OUT    = OUT_DIR / "crate_lrp_subspaces.png"


# ---------------------------------------------------------------------------
# Model setup (identical to basketball_montage.py)
# ---------------------------------------------------------------------------

def add_vgg16_aliases(model):
    feat = list(model.features)
    block_conv_counts = [2, 2, 3, 3, 3]
    ptr = 0
    for b, n_convs in enumerate(block_conv_counts, start=1):
        for c in range(1, n_convs + 1):
            object.__setattr__(model.features, f"conv{b}_{c}", feat[ptr])
            if ptr + 1 < len(feat):
                object.__setattr__(model.features, f"relu{b}_{c}", feat[ptr + 1])
            ptr += 2
        object.__setattr__(model.features, f"pool{b}", feat[ptr])
        ptr += 1
    return model


def load_model():
    model = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
    )
    model = add_vgg16_aliases(model)

    mean = constants.IMAGENET_MEAN
    std  = constants.IMAGENET_STD

    # Carton images need resize + crop; use this as the "canonical" transform
    # stored on the model (required by VGGLRPExplainer).
    rc_transform         = T.Compose([T.Resize(256), T.CenterCrop(224)])
    carton_input_transform = T.Compose([
        rc_transform,
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ])
    # Crate images are already 224×224; only need tensor + normalise.
    crate_input_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ])

    setattr(model, models.ATTRIBUTE_TRANSFORMATION, (rc_transform, carton_input_transform))
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, rc_transform, carton_input_transform, crate_input_transform


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_montage(model, explainer, input_transform,
                 img_paths, target_class, gb_inspector,
                 title, out_path, display_transform=None,
                 subspace_labels=None, fs_title=9, fs_ylabel=7,
                 show_suptitle=True, show_ylabels=True):
    """
    display_transform: optional callable (PIL -> PIL) applied before imshow.
                       If None, the image is shown as-is (already 224x224).
    subspace_labels:   list of K strings for subspace column titles (default: "Subspace {s}").
    fs_title:          font size for column titles.
    fs_ylabel:         font size for row ylabels.
    show_suptitle:     if False, suppress the figure suptitle.
    show_ylabels:      if False, suppress row ylabel (image filename).
    """

    nrows = len(img_paths)
    ncols = 2 + K
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, img_path in enumerate(img_paths):
        img = putils.load_image(str(img_path))
        x   = input_transform(img).to(DEVICE)

        logits, standard_heatmap, info = explainer.explain_with_inspector(
            x, target_class, inspector=gb_inspector, top_k=K,
        )

        # Reorder subspace heatmaps to canonical 0..K-1 order
        canonical_order = np.argsort(info.top_k_sources)
        sub_hm = info.input_top_k_source_heatmaps[canonical_order]  # (K, C, H, W)
        sub_hm = sub_hm.sum(axis=1)                                  # (K, H, W)
        std_hm = standard_heatmap.sum(axis=0)                        # (H, W)

        # Image column
        display_img = display_transform(img) if display_transform is not None else img
        plt.sca(axes[row_idx, 0])
        putils.viz.imshow(display_img)
        if show_ylabels:
            axes[row_idx, 0].set_ylabel(img_path.name, fontsize=fs_ylabel)
        axes[row_idx, 0].set_xlabel('')
        if row_idx == 0:
            axes[row_idx, 0].set_title("Image", fontsize=fs_title)

        # Standard LRP column
        plt.sca(axes[row_idx, 1])
        putils.viz.heatmap(std_hm,
                           title="LRP" if row_idx == 0 else "")
        axes[row_idx, 1].set_xlabel('')
        if row_idx == 0:
            axes[row_idx, 1].set_title("LRP", fontsize=fs_title)

        # Per-subspace columns — normalise to the same scale as the full LRP
        # heatmap so that the contribution of each subspace is visually comparable.
        for s in range(K):
            col_title = (subspace_labels[s] if subspace_labels is not None
                         else f"Subspace {s}")
            plt.sca(axes[row_idx, 2 + s])
            putils.viz.heatmap(sub_hm[s],
                               title=col_title if row_idx == 0 else "",
                               reference_heatmap=std_hm)
            axes[row_idx, 2 + s].set_xlabel('')
            if row_idx == 0:
                axes[row_idx, 2 + s].set_title(col_title, fontsize=fs_title)

    if show_suptitle:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[device] {DEVICE}")

    model, rc_transform, carton_input_transform, crate_input_transform = load_model()
    explainer = factory.make_explainer("lrp", model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in [
        dict(
            img_paths         = CARTON_IMGS,
            target            = CARTON_TARGET,
            u_path            = CARTON_U,
            out_path          = CARTON_OUT,
            title             = f"VGG16 ImageNet · logit {CARTON_TARGET} (carton) · layer {LAYER}",
            input_transform   = carton_input_transform,
            display_transform = rc_transform,   # needs resize+crop before display
            subspace_labels   = ["1", "2", "3: Writing on box", "4: Watermark"],
            fs_title          = 22,
            fs_ylabel         = 16,
            show_suptitle     = False,
        ),
        dict(
            img_paths         = CRATE_IMGS,
            target            = CRATE_TARGET,
            u_path            = CRATE_U,
            out_path          = CRATE_OUT,
            title             = f"VGG16 ImageNet · logit {CRATE_TARGET} (crate) · layer {LAYER}",
            input_transform   = crate_input_transform,
            display_transform = None,           # already 224×224
            subspace_labels   = ["1", "2: Watermark", "3", "4: Writing on box"],
            fs_title          = 22,
            fs_ylabel         = 16,
            show_suptitle     = False,
            show_ylabels      = False,
        ),
    ]:
        print(f"\n[{cfg['out_path'].stem}] target={cfg['target']}  U={cfg['u_path'].name}")

        U_raw = torch.load(str(cfg['u_path']), map_location="cpu")
        assert U_raw.shape == (512, 512), f"Unexpected U shape: {U_raw.shape}"
        ss = U_raw.shape[1] // K
        U  = U_raw.reshape(512, K, ss).float().to(DEVICE)
        print(f"  U reshaped to {U.shape}  (nd=512, k={K}, ss={ss})")

        gb_inspector = inspector.GroupBasisInspector(layer=LAYER, weights=U).to(DEVICE)

        make_montage(
            model, explainer,
            input_transform   = cfg['input_transform'],
            img_paths         = cfg['img_paths'],
            target_class      = cfg['target'],
            gb_inspector      = gb_inspector,
            title             = cfg['title'],
            out_path          = cfg['out_path'],
            display_transform = cfg['display_transform'],
            subspace_labels   = cfg.get('subspace_labels'),
            fs_title          = cfg.get('fs_title', 9),
            fs_ylabel         = cfg.get('fs_ylabel', 7),
            show_suptitle     = cfg.get('show_suptitle', True),
            show_ylabels      = cfg.get('show_ylabels', True),
        )


if __name__ == "__main__":
    main()
