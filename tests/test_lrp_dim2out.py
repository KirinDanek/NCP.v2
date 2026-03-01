# VGG16_BN + custom LRP + DRSA subspaces at conv4_3 (features.22)
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from matplotlib import pyplot as plt

# --------------------- EDIT THESE ---------------------
CKPT_PATH      = '/n/fs/ncp/NCP.v2/data/trained_models/vgg16_scratch_cf90A.pt'  # or None to use random init
IMAGE_FILEPATH = '/n/fs/ncp/NCP.v2/data/images/waterbirds_out/waterbirds_cf90_A/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg'
U_PATH         = '/n/fs/ncp/drsa-demo/notebooks/U_waterbirds.npy'   # expected shape (256, K)
TARGET_CLASS   = 1     # 1 = waterbird
SAVE_DIR       = Path('/n/fs/ncp/drsa-demo/notebooks/tmp')
OUT_BASENAME   = 'waterbirds_vggbn_lrp_drsa'  # base name for saved plots
# -----------------------------------------------------

SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Make custom lrp.py visible
sys.path.insert(0, '/n/fs/ncp/NCP.v2')
import lrp as lrp_mod  # your /n/fs/ncp/NCP.v2/lrp.py
lrp = lrp_mod.lrp

# --- Globals for flatten shape ---
FLATTEN_IN_SHAPE = None

# --------------------- Preprocessing ---------------------
rc_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load_image(path: str):
    return Image.open(path).convert('RGB')

# --------------------- Model loading ---------------------
# Load U (supports (C,K) or (C,S,D))
def load_U(U_PATH, device):
    if U_PATH.endswith('.npy'):
        U_np = np.load(U_PATH)
        U_t = torch.from_numpy(U_np).float()
    else:
        obj = torch.load(U_PATH, map_location='cpu')
        if isinstance(obj, dict):            # allow common keys
            for key in ['U', 'weights', 'basis']:
                if key in obj:
                    obj = obj[key]
                    break
        if isinstance(obj, np.ndarray):
            U_t = torch.from_numpy(obj).float()
        elif torch.is_tensor(obj):
            U_t = obj.float()
        else:
            raise ValueError(f"Unsupported U container type: {type(obj)}")
    return U_t.to(device)

def load_vgg16_bn():
    model = models.vgg16_bn(weights=None)
    if CKPT_PATH and os.path.isfile(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        # Extract state_dict robustly
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                sd = ckpt['state_dict']
            elif 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
                sd = ckpt['model_state_dict']
            elif 'model' in ckpt and isinstance(ckpt['model'], dict):
                sd = ckpt['model']
            else:
                sd = ckpt
        else:
            sd = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt
        # Strip 'module.' if DDP
        clean_sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
        # Resize head if needed
        if 'classifier.6.weight' in clean_sd:
            out_feats = clean_sd['classifier.6.weight'].shape[0]
            if model.classifier[6].out_features != out_feats:
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_feats)
        missing, unexpected = model.load_state_dict(clean_sd, strict=False)
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if missing: print("  missing (first 10):", missing[:10])
        if unexpected: print("  unexpected (first 10):", unexpected[:10])
    model.eval()
    return model.to(DEVICE)

# --------------------- Hooks ---------------------
def register_hooks(model: nn.Module):
    def save_io(mod, inp, out):
        mod.input = inp[0].detach()
        mod.output = out.detach()
    for m in model.modules():
        m.register_forward_hook(save_io)

# --------------------- Linearize for propagation ---------------------
class _Flatten(nn.Module):
    def forward(self, x): return torch.flatten(x, 1)

def linearize(model: nn.Module):
    layers = []
    for m in model.features: layers.append(m)
    layers.append(model.avgpool)
    layers.append(_Flatten())
    for m in model.classifier: layers.append(m)
    return layers

# --------------------- LRP dispatch (wrap your lrp for extra types) ---------------------
def lrp_dispatch(module: nn.Module, R: torch.Tensor):
    with torch.no_grad():
        # Identity-ish layers
        if isinstance(module, (nn.ReLU, nn.Dropout, nn.BatchNorm2d)):
            return R

        # Flatten: reshape using cached shape from avgpool output
        if isinstance(module, _Flatten):
            if FLATTEN_IN_SHAPE is None:
                raise RuntimeError("FLATTEN_IN_SHAPE not set. Run a forward pass and set from avgpool.output.")
            return R.view(FLATTEN_IN_SHAPE)

        # AdaptiveAvgPool2d: simple redistribution (nearest-upsample then z-rule style)
        if isinstance(module, nn.AdaptiveAvgPool2d):
            Hin, Win = module.input.shape[-2:]
            Rout = F.interpolate(R, size=(Hin, Win), mode='nearest')
            return module.input * (Rout / (module.input.abs() + 1e-9))

        # AvgPool2d: same approach
        if isinstance(module, nn.AvgPool2d):
            Hin, Win = module.input.shape[-2:]
            Rout = F.interpolate(R, size=(Hin, Win), mode='nearest')
            return module.input * (Rout / (module.input.abs() + 1e-9))

        # Hand off to your custom LRP for the core layers
        if isinstance(module, (nn.MaxPool2d, nn.Conv2d, nn.Linear, nn.LogSoftmax)):
            if isinstance(module, nn.Conv2d) and module is FIRST_CONV:
                return lrp(module, R, lrp_var='first')
            return lrp(module, R, lrp_var='alpha', param=1.0)

        # Default: no-op
        return R

# --------------------- Helpers to find conv4_3 (features.22) ---------------------
def find_layer_index(layers, model: nn.Module, ref: str = "features.22"):
    # map name->module
    name_to_mod = {name: mod for name, mod in model.named_modules()}
    if ref not in name_to_mod:
        raise ValueError(f"Layer '{ref}' not found.")
    tgt = name_to_mod[ref]
    for i, m in enumerate(layers):
        if m is tgt:
            return i
    raise ValueError(f"Layer '{ref}' not in linearized layers.")

# --------------------- Propagation utilities ---------------------
def propagate_from_top_to_index(layers, R, stop_at_idx: int):
    # reverse from last layer until we are right above stop_at_idx
    for i in reversed(range(len(layers))):
        if i == stop_at_idx:
            break
        R = lrp_dispatch(layers[i], R)
    return R  # relevance at output of layers[stop_at_idx]

def propagate_from_index_to_input(layers, R, start_idx: int):
    for i in reversed(range(start_idx + 1)):
        R = lrp_dispatch(layers[i], R)
    return R  # relevance at input (B,3,H,W)

# --------------------- DRSA split at conv4_3 ---------------------
def split_relevance_by_subspace(R_target: torch.Tensor, U: torch.Tensor):
    """
    R_target: (1, C, H, W) relevance at target layer (conv4_3)
    U:  (C, K)   -> K one-dimensional bases  -> returns K tensors
        (C, S, D)-> S subspaces of D dims    -> returns S tensors
    """
    with torch.no_grad():
        B, C, H, W = R_target.shape
        assert B == 1, "Batch=1 expected"
        Rchw = R_target.view(C, H * W)  # (C, HW)

        if U.dim() == 2:
            # U: (C, K) -> per-basis projection
            K = U.shape[1]
            coeffs = U.t().mm(Rchw)    # (K, HW)
            out = [(U[:, [k]].mm(coeffs[[k], :])).view(1, C, H, W) for k in range(K)]
            return out  # length K

        elif U.dim() == 3:
            # U: (C, S, D) -> per-subspace projection (block projector)
            C_u, S, D = U.shape
            assert C_u == C, f"U first dim {C_u} != channels {C}"
            out = []
            # optional: orthonormalize each block if not already orthonormal
            # for s in range(S): U[:, s, :], _ = torch.linalg.qr(U[:, s, :], mode='reduced')
            for s in range(S):
                Us = U[:, s, :]               # (C, D)
                coeffs = Us.t().mm(Rchw)      # (D, HW)
                Rk = Us.mm(coeffs).view(1, C, H, W)  # (1, C, H, W)
                out.append(Rk)
            return out  # length S

        else:
            raise ValueError(f"U must be 2D or 3D; got {U.shape}")


# --------------------- Visualization ---------------------
def save_heatmap(heatmap_2d: np.ndarray, out_png: Path, title: str = ""):
    # symmetric percentile scaling
    p = 99
    vmax = np.percentile(np.abs(heatmap_2d), p)
    vmin = -vmax
    plt.figure(figsize=(5,5))
    plt.imshow(heatmap_2d, cmap='seismic', vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title: plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SAVED] {out_png}")

# --------------------- Main ---------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Model + hooks
    model = load_vgg16_bn()
    register_hooks(model)
    FIRST_CONV = model.features[0]  # for special LRP treatment
    layers = linearize(model)
    target_idx = find_layer_index(layers, model, "features.30")  # conv4_3 in vgg16_bn
    name_to_mod = {n: m for n, m in model.named_modules()}
    tmod = name_to_mod["features.30"]
    print("Target layer:", tmod.__class__.__name__, "out_channels =", getattr(tmod, "out_channels", None))

    # Load U (C,K)
    U = load_U(U_PATH, DEVICE)

    # Single image
    pil_img = load_image(IMAGE_FILEPATH)
    x = input_transform(pil_img).unsqueeze(0).to(DEVICE)
    disp_img = rc_transform(pil_img)

    # Forward to populate hooks
    out = model(x)
    # Cache flatten input shape from avgpool.output
    FLATTEN_IN_SHAPE = model.avgpool.output.shape  # e.g., (1,512,7,7)

    # Initialize output relevance at TARGET_CLASS
    R = torch.zeros_like(out)
    R[0, TARGET_CLASS] = out[0, TARGET_CLASS]

    # Relevance at conv4_3 output
    R_at_target = propagate_from_top_to_index(layers, R, target_idx)

    C_target = R_at_target.shape[1]
    if U.dim() == 2:
        assert U.shape[0] == C_target, f"U first dim {U.shape[0]} != channels {C_target}"
        N_SUB = U.shape[1]  # number of 1-D bases
    elif U.dim() == 3:
        assert U.shape[0] == C_target, f"U first dim {U.shape[0]} != channels {C_target}"
        N_SUB = U.shape[1]  # number of subspaces (blocks)
    else:
        raise ValueError(f"U must be 2D or 3D, got {U.shape}")

    # Standard input-space LRP (propagate all the way down)
    R_input = propagate_from_index_to_input(layers, R_at_target.clone(), start_idx=target_idx)
    standard_heat = R_input.squeeze(0).sum(dim=0).detach().cpu().numpy()
    save_heatmap(standard_heat, SAVE_DIR / f"{OUT_BASENAME}_standard.png", title="Standard LRP (sum over RGB)")

    # DRSA subspaces at conv4_3
    with torch.no_grad():
        # Ensure channel dimension matches conv4_3 (256 for VGG16-BN)
        C_target = R_at_target.shape[1]
        if U.shape[0] != C_target:
            raise ValueError(f"U first dim ({U.shape[0]}) != conv4_3 channels ({C_target})")

    Rk_list = split_relevance_by_subspace(R_at_target, U)  # list length = N_SUB
    for idx, Rk in enumerate(Rk_list, start=1):
        Rk_input = propagate_from_index_to_input(layers, Rk, start_idx=target_idx)
        heat_k = Rk_input.squeeze(0).sum(dim=0).detach().cpu().numpy()
        save_heatmap(heat_k, SAVE_DIR / f"{OUT_BASENAME}_subspace_{idx:02d}.png", title=f"DRSA Subspace {idx}")

    # Also save the input image for reference
    plt.figure(figsize=(5,5)); plt.imshow(disp_img); plt.axis('off'); plt.tight_layout()
    img_path = SAVE_DIR / f"{OUT_BASENAME}_image.png"
    plt.savefig(img_path, dpi=200, bbox_inches='tight', pad_inches=0); plt.close()
    print(f"[SAVED] {img_path}")
