import torch
import numpy as np
from torch import nn
from lrp import *
import glob 
import os
import re

from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt



### vars
MODEL_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/results/pruned-models/aug-basketball-80.pth'
IMAGE_DIR_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images/'
OUTPUT_HEATMAP_PATH = '/u/kd9132/n/fs/ncp/NCP.v2/results/pruned-models/80-basketball-ablated/ncp.png'

# "basketball" in IN bball binary dataset
TARGET_CLASS = 0


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
    heatmap_seismic = 10*((np.abs(heatmap)**3.0).mean()**(1.0/3))
    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *=0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure()
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(heatmap, cmap=my_cmap, vmin=-heatmap_seismic, vmax=heatmap_seismic, interpolation='nearest')
    plt.savefig(out_path)
    plt.close()

## takes reverse index of module
def layer_to_gamma_param(module_idx: int) -> float:
    if module_idx <= 6:                         # classifier layers
        return 0.01 # or 0.0
    elif 7 <= module_idx <= 13:                 # Conv5
        return 0.01 # or 0.0
    elif 14 <= module_idx <= 20:                # Conv4
        return 0.10 # 0.10
    elif 21 <= module_idx <= 27:                # Conv3
        return 0.25 # 0.25
    else:
        if module_idx < 28 or module_idx > 37:
            print(f'unexpected module index {module_idx}')                                       # Conv2, Conv1, and anything earlier
        return 0.50


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    pattern = re.compile(r'img-(\d+)\.jpg')

    # Load checkpoint
    checkpoint = torch.load(MODEL_FILEPATH, map_location=device)
    model = checkpoint['model'].to(device)
    model.eval()

    register_hooks(model)

    # Flatten model into an ordered list
    modules = list(model.before) + list(model.after) + list(model.classifier)

    image_paths = glob.glob(os.path.join(IMAGE_DIR_FILEPATH, 'img-*.jpg'))


    for image_path in image_paths:
        ### 3. Load and preprocess the input image
        input_tensor, _ = load_and_preprocess(image_path, device=device)

        with torch.no_grad():
            output = model(input_tensor)
            R = torch.zeros_like(output)
            R[0, TARGET_CLASS] = output[0, TARGET_CLASS]

        # Debug info
        print("Initial output relevance:", R.sum())

        ### 4. Compute LRP attributions for the fixed TARGET_CLASS
        # Try different LRP rules for comparison
        lrp_rules = [
            ('alphabeta', 1.0),     # alpha=1, beta=0 
            ('gamma', 'heuristic'),       
        ]
        
        for rule_name, param in lrp_rules:
            print(f"\n=== LRP rule: {rule_name} with param {param} ===")
            R_test = R.clone()
            
            if rule_name == 'gamma' and param == 'heuristic':
                for i, module in enumerate(reversed(modules)):
                    if i == len(modules) - 1: 
                        R_test = lrp(module, R_test, lrp_var='first')
                        print(f"handle pixel layer at idx {i}")

                    else:
                        dynamic_param = layer_to_gamma_param(i)
                        R_test = lrp(module, R_test, lrp_var=rule_name, param=dynamic_param)
                    print(f"After layer {i} ({module.__class__.__name__}): R min={R_test.min().item():.2f}, max={R_test.max().item():.2f}, sum={R_test.sum().item():.2f}")


            else: 
                # Propagate in reverse order
                for i, module in enumerate(reversed(modules)):
                    if i == len(modules) - 1:
                        R_test = lrp(module, R_test, lrp_var='first')
                        print(f"handle pixel layer at idx {i}")
                    else:               
                        R_test = lrp(module, R_test, lrp_var=rule_name, param=param)
                    print(f"After layer {i} ({module.__class__.__name__}): R min={R_test.min().item():.2f}, max={R_test.max().item():.2f}, sum={R_test.sum().item():.2f}")
                    
            print(f"Final input relevance sum: {R_test.sum().item():.4f}")

            # Save heatmap for this rule and image
            filename = os.path.basename(image_path)
            match = pattern.match(filename)
            img_idx = match.group(1) # image index (eg, img-1.jpg)
            rule_output_path = OUTPUT_HEATMAP_PATH.replace('.png', f'_{rule_name}_img-{img_idx}.png')
            visualize_and_save_lrp(R_test, out_path=rule_output_path)
