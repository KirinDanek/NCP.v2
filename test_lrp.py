import torch
import numpy as np
import AugmentedVGG16
from torch import nn
from AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix
from lrp import *

from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt



### vars
SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = []  # test: ablate "ball" subspace (ix 3). Should be easy to see in LRP heatmap
U_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt'
IMAGE_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa-basketball-img3.jpg'
OUTPUT_HEATMAP_PATH = 'lrp_heatmap.png'

# "basketball" in ImageNet is class index 430 (zero‐indexed)
TARGET_CLASS = 430


def load_and_preprocess(image_path: str, device: torch.device):
    """
    Load an image from disk, resize→center‐crop→tensor→normalize for VGG16.
    Returns (preprocessed_tensor, original_PIL).
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)  # shape: (1,3,224,224)
    return tensor, img

### hooks for augmented vgg16
def register_hooks(model):
    def save_input_output(mod, inp, out):
        mod.input = inp[0].detach().clone()
        mod.output = out.detach().clone()

    for module in model.modules():
        module.register_forward_hook(save_input_output)


def visualize_and_save_lrp(attribution_tensor: torch.Tensor,
                           out_path: str = OUTPUT_HEATMAP_PATH):
    """
    Save LRP heatmap with improved processing for better visualization.
    """
    # Move to CPU and convert to numpy
    attr = attribution_tensor.squeeze(0).cpu().detach().numpy()  # → (3, 224, 224)
    
    if attr.shape != (3, 224, 224):
        raise ValueError(f"Expected attribution_tensor of shape (1, 3, 224, 224), got {attr.shape}")
    
    # Sum across RGB channels to get spatial heatmap
    heatmap = attr.sum(axis=0)  # → (224, 224)
    np.save(out_path.replace('.png', '_heatmap.npy'), heatmap)
    print(f"Raw heatmap stats — min: {heatmap.min():.6f}, max: {heatmap.max():.6f}, mean: {heatmap.mean():.6f}")
    
    # Method 1: Positive relevance only (your original approach)
    heatmap_pos = np.maximum(heatmap, 0)
    p99 = np.percentile(heatmap_pos, 94)
    heatmap_pos = np.clip(heatmap_pos, 0, p99)

    #max_val_pos = heatmap_pos.max()
    
    if p99 > 0:
        heatmap_pos_norm = heatmap_pos / p99
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap_pos_norm, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path.replace('.png', '_positive_only.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Positive-only heatmap saved to '{out_path.replace('.png', '_positive_only.png')}'")
    
    # Method 2: Absolute values (recommended)
    #heatmap_abs = np.abs(heatmap)
    #max_val_abs = heatmap_abs.max()
    
    #if max_val_abs > 0:
    #    heatmap_abs_norm = heatmap_abs / max_val_abs
    #    plt.figure(figsize=(8, 8))
    #    plt.imshow(heatmap_abs_norm, cmap='hot')
    #    plt.axis('off')
    #    plt.tight_layout()
    #    plt.savefig(out_path.replace('.png', '_absolute.png'), bbox_inches='tight', pad_inches=0)
     #   plt.close()
    #    print(f"Absolute value heatmap saved to '{out_path.replace('.png', '_absolute.png')}'")

    # Method 3: Centered around zero with diverging colormap (outlier-protected)
    # This shows both positive (red) and negative (blue) contributions
    heatmap_centered = heatmap.copy()
    # Protect against outliers using percentile clipping
    #pos_p99 = np.percentile(heatmap_centered[heatmap_centered > 0], 99) if np.any(heatmap_centered > 0) else 0
    #neg_p99 = np.percentile(np.abs(heatmap_centered[heatmap_centered < 0]), 99) if np.any(heatmap_centered < 0) else 0

    # Use the larger of the two percentiles for symmetric clipping
    #clip_val = max(pos_p99, neg_p99)
    clip_val = np.percentile(np.abs(heatmap), 97)

    if clip_val > 0:
    ## Clip outliers symmetrically
        heatmap_centered_clipped = np.clip(heatmap_centered, -clip_val, clip_val)
   ### 
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap_centered_clipped, cmap='RdBu_r', vmin=-clip_val, vmax=clip_val)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path.replace('.png', '_centered.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Centered (outlier-protected) heatmap saved to '{out_path.replace('.png', '_centered.png')}'")


    
    # Method 4: Percentile-based normalization (often works best)
    # This handles outliers better
    #p99 = np.percentile(np.abs(heatmap), 99)
    #heatmap_clipped = np.clip(np.abs(heatmap), 0, p99)
    #heatmap_norm = heatmap_clipped / p99 if p99 > 0 else heatmap_clipped
   ## 
    #plt.figure(figsize=(8, 8))
    #plt.imshow(heatmap_norm, cmap='hot')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig(out_path.replace('.png', '_pbn.png'), bbox_inches='tight', pad_inches=0)
    #plt.close()
    #print(f"Percentile-normalized heatmap saved to '{out_path.replace('.png', '_pbn.png')}'")

def get_augmented_vgg16_lrp_param(module_idx: int) -> float:
    """
    γ-schedule for LRP-γ on AugmentedVGG16, counting *from the output side* as we
    iterate through reversed(modules).

    ── classifier head ─────────────── 0.00
    ── Conv5 block  ─────────────────  0.00
    ── Augmented 1×1 + Conv4 block ─  0.10
    ── Conv3 block  ─────────────────  0.25
    ── Conv2 + Conv1 blocks ─────────  0.50  (all remaining layers)
    """
    if module_idx <= 6:                         # classifier layers
        return 0.0
    elif 7 <= module_idx <= 13:                 # Conv5
        return 0.0
    elif 14 <= module_idx <= 22:                # 1×1 augmented + Conv4
        return 0.10
    elif 23 <= module_idx <= 29:                # Conv3
        return 0.25
    else:     
        if module_idx < 30 or module_idx > 39:
            print(f'unexpected module index {module_idx}') 
                                            # Conv2, Conv1, and anything earlier
        return 0.50



if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### 1. Load tensor U, ablate and move to GPU
    U = torch.load(U_FILEPATH)  # shape: (512, 512)
    U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)
    #U_ab = torch.eye(512)
    #U_ab_T = torch.eye(512)


    U_ab = U_ab.to(device)
    U_ab_T = U_ab_T.to(device)

    ### 2. Build the augmented model, send to GPU, set eval()
    augmentedVGG16 = AugmentedVGG16(U=U_ab, UT=U_ab_T).to(device)
    augmentedVGG16.eval()

    register_hooks(augmentedVGG16)

    ### 3. Load and preprocess the input image
    input_tensor, _ = load_and_preprocess(IMAGE_FILEPATH, device=device)

    with torch.no_grad():
        output = augmentedVGG16(input_tensor)
        R = torch.zeros_like(output)
        R[0, TARGET_CLASS] = output[0, TARGET_CLASS]

    # Debug info
    assert any(hasattr(m, "input") for m in augmentedVGG16.modules()), "Forward hook registration failed"
    print("Encode weight NaNs:", torch.isnan(augmentedVGG16.encode.weight).any().item())
    print("Decode weight NaNs:", torch.isnan(augmentedVGG16.decode.weight).any().item())
    print("Initial output relevance:", R.sum())

    ### 4. Compute LRP attributions for the fixed TARGET_CLASS
    # Flatten model into an ordered list
    modules = list(augmentedVGG16.before) + [augmentedVGG16.encode, augmentedVGG16.decode] + list(augmentedVGG16.after) + list(augmentedVGG16.classifier)

    # Try different LRP rules for comparison
    lrp_rules = [
        #('epsilon', 1e-2),      # epsilon rule - often good baseline
        #('alphabeta', 2.0),     # alpha=2, beta=-1 (more aggressive)
        #('alphabeta', 1.0),     # alpha=1, beta=0 (your original)
        #('gamma', 'heuristic'),        # gamma rule
        ('gamma', 0.0)
    ]
    
    for rule_name, param in lrp_rules:
        print(f"\n=== LRP rule: {rule_name} with param {param} ===")
        R_test = R.clone()
        
        if rule_name == 'gamma' and param == 'heuristic':
            for i, module in enumerate(reversed(modules)):
                if True: # i == 16
                    #R_test = lrp(module, R_test, lrp_var='epsilon', param=20.0)
                #else:
                    
                    dynamic_param = get_augmented_vgg16_lrp_param(i)
                    print(f"Gamma heuristic module = {module} with gamma = {dynamic_param}")
                    R_test = lrp(module, R_test, lrp_var=rule_name, param=dynamic_param)
                print(f"After layer {i} ({module.__class__.__name__}): R min={R_test.min().item():.2f}, max={R_test.max().item():.2f}, sum={R_test.sum().item():.2f}")


        else: 
            # Propagate in reverse order
            for i, module in enumerate(reversed(modules)):
                              
                R_test = lrp(module, R_test, lrp_var=rule_name, param=param)
                if i == 15:
                    l15 = R_test
                if i == 17:
                    l17 = R_test
                print(f"After layer ({module.__class__.__name__}): R min={R_test.min().item():.2f}, max={R_test.max().item():.2f}, sum={R_test.sum().item():.2f}")
                # Check for issues during propagation
                if torch.isnan(R_test).any():
                    print(f"ERROR: NaN detected after {module.__class__.__name__}")
                    break
                if torch.isinf(R_test).any():
                    print(f"ERROR: Inf detected after {module.__class__.__name__}")
                    break
                
        print(f"Final input relevance sum: {R_test.sum().item():.4f}")

        abs_diff = abs(l15 - l17)
        print(torch.amax(abs_diff), " ", torch.amin(abs_diff))
        
        # Save heatmap for this rule
        rule_output_path = OUTPUT_HEATMAP_PATH.replace('.png', f'_{rule_name}_{param}.png')
        visualize_and_save_lrp(R_test, out_path=rule_output_path)
