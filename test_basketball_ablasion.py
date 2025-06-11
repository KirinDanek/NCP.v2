import torch
import numpy as np
import NCP
from torch import nn
from NCP import AugmentedVGG16, ablate_subspace_matrix
from lrp import *

from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt



### vars
SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = [3]  # test: ablate "ball" subspace (ix 3). Should be easy to see in LRP heatmap
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
        mod.input = inp[0]
        mod.output = out

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
    
    print(f"Raw heatmap stats — min: {heatmap.min():.6f}, max: {heatmap.max():.6f}, mean: {heatmap.mean():.6f}")
    
    
    # Method 4: Percentile-based normalization (often works best)
    # This handles outliers better
    #p99 = np.percentile(np.abs(heatmap), 99)
    #heatmap_clipped = np.clip(np.abs(heatmap), 0, p99)
    #heatmap_norm = heatmap_clipped / p99 if p99 > 0 else heatmap_clipped
    
    plt.figure(figsize=(8, 8))
    #plt.imshow(heatmap_norm, cmap='hot')
    plt.imshow(heatmap, cmap='RdBu_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Percentile-normalized heatmap saved to '{out_path}'")


if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### 1. Load tensor U, ablate and move to GPU
    U = torch.load(U_FILEPATH)  # shape: (512, 512)
    U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)
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
        #('epsilon', 1e-6),      # epsilon rule - often good baseline
        #('alphabeta', 2.0),     # alpha=2, beta=-1 (more aggressive)
        ('alphabeta', 1),     # alpha=1, beta=0 (your original)
        #('gamma', 0.25),        # gamma rule
    ]
    
    for rule_name, param in lrp_rules:
        print(f"\n=== LRP rule: {rule_name} with param {param} ===")
        R_test = R.clone()
        
        # Propagate in reverse order
        for module in reversed(modules):
            R_test = lrp(module, R_test, lrp_var=rule_name, param=param)
            
            # Check for issues during propagation
            if torch.isnan(R_test).any():
                print(f"ERROR: NaN detected after {module.__class__.__name__}")
                break
            if torch.isinf(R_test).any():
                print(f"ERROR: Inf detected after {module.__class__.__name__}")
                break
                
        print(f"Final input relevance sum: {R_test.sum().item():.4f}")
        
        # Save heatmap for this rule
        rule_output_path = OUTPUT_HEATMAP_PATH.replace('.png', f'_{rule_name}_{param}.png')
        visualize_and_save_lrp(R_test, out_path=rule_output_path)
