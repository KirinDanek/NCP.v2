import torch
import numpy as np
from torch import nn
from lrp import *

from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt

from torchvision.models import vgg16



### vars
IMAGE_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa-basketball-img3.jpg'
OUTPUT_HEATMAP_PATH = 'lrp_heatmap_van.png'

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
    
    # Sum across RGB channels to get spatial heatmap
    heatmap = attr.sum(axis=0)  # → (224, 224)
    
    print(f"Raw heatmap stats — min: {heatmap.min():.6f}, max: {heatmap.max():.6f}, mean: {heatmap.mean():.6f}")
    
    # Method 1: Positive relevance only (your original approach)
    heatmap_pos = np.maximum(heatmap, 0)
    #p99 = np.percentile(heatmap_pos, 94)
    #heatmap_pos = np.clip(heatmap_pos, 0, p99)
    
    if True:
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap_pos, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path.replace('.png', '_positive_only.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Positive-only heatmap saved to '{out_path.replace('.png', '_positive_only.png')}'")
    

    # Method 3: Centered around zero with diverging colormap (outlier-protected)
    # This shows both positive (red) and negative (blue) contributions
    heatmap_centered = heatmap.copy()
    # Protect against outliers using percentile clipping
    #pos_p99 = np.percentile(heatmap_centered[heatmap_centered > 0], 99) if np.any(heatmap_centered > 0) else 0
    #neg_p99 = np.percentile(np.abs(heatmap_centered[heatmap_centered < 0]), 99) if np.any(heatmap_centered < 0) else 0

    # Use the larger of the two percentiles for symmetric clipping
    #clip_val = max(pos_p99, neg_p99)
    #clip_val = p99
    #clip_val = 0.3

    if True:
    # Clip outliers symmetrically
        #heatmap_centered_clipped = np.clip(heatmap_centered, -0.02, 0.02)
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap_centered, cmap='bwr')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path.replace('.png', '_centered.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Centered heatmap saved to '{out_path.replace('.png', '_centered.png')}'")
    heatmap_seismic = 10*((np.abs(heatmap)**3.0).mean()**(1.0/3))
    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *=0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure()
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(heatmap, cmap=my_cmap, vmin=-heatmap_seismic, vmax=heatmap_seismic, interpolation='nearest')
    plt.savefig(out_path.replace('.png', '_seismic.png'))
    plt.close()

def get_vgg16_lrp_param(module_idx: int) -> float:
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
        return 0.0 # 0.0
    elif 7 <= module_idx <= 13:                 # Conv5
        return 0.0 #0.0
    elif 14 <= module_idx <= 20:                # 1×1 augmented + Conv4
        return 0.10 # 0.10
    elif 21 <= module_idx <= 27:                # Conv3
        return 0.25 # 0.25
    else:
        if module_idx < 28 or module_idx > 37:
            print(f'unexpected module index {module_idx}')                                       # Conv2, Conv1, and anything earlier
        return 0.50


if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ### 2. Build the model, send to GPU, set eval()
    model = vgg16(pretrained=True).to(device)
    model.eval()

    register_hooks(model)

    ### 3. Load and preprocess the input image
    input_tensor, _ = load_and_preprocess(IMAGE_FILEPATH, device=device)

    #input_tensor.requires_grad_(True) ### debug remove
    with torch.no_grad():
        output = model(input_tensor)
        #predicted_class = output.argmax(dim=1).item()
        R = torch.zeros_like(output)
        R[0, TARGET_CLASS] = output[0, TARGET_CLASS]

    assert any(hasattr(m, "input") for m in model.modules()), "Forward hook registration failed"
    print("Initial output relevance:", R.sum())

    ### 4. Compute LRP attributions for the fixed TARGET_CLASS
    # Flatten model into an ordered list
    modules = list(model.features) + list(model.classifier)

    # Try different LRP rules for comparison
    lrp_rules = [
        #('simple', 1e-6),      # epsilon rule - often good baseline
        #('alphabeta', 2),     # alpha=2, beta=-1 (more aggressive)
        ('alphabeta', 1.0),     # alpha=1, beta=0 (your original)
        ('gamma', 0.0),        # gamma rule
        ('gamma', 0.25),
        ('gamma', 'heuristic')
    ]

    for rule_name, param in lrp_rules:
        print(f"\n=== LRP rule: {rule_name} with param {param} ===")
        R_test = R.clone()
        
        if rule_name == 'gamma' and param == 'heuristic':
            for i, module in enumerate(reversed(modules)):
                if i == 37:
                    R_test = lrp(module, R_test, lrp_var='first')
                    print(f"idx {i}: handling pixel layer")
                else:
                    
                    dynamic_param = get_vgg16_lrp_param(i)
                    print(f"Gamma heuristic module = {module} at index {i} with gamma = {dynamic_param}")
                    R_test = lrp(module, R_test, lrp_var=rule_name, param=dynamic_param)
                print(f"R stats at layer {i}: R min={R_test.min().item():.2f}, max={R_test.max().item():.2f}, sum={R_test.sum().item():.2f}") 
        else: 
            # Propagate in reverse order
            for i, module in enumerate(reversed(modules)):
                if i == 37:
                    R_test = lrp(module, R_test, lrp_var='first')
                    print(f"idx {i}: handling pixel layer")
                else:

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
