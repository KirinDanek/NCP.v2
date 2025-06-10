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
IRRELEVANT_SUBSPACES = []  # test: ablate "ball" subspace (ix 3). Should be easy to see in LRP heatmap
U_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt'
IMAGE_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa-basketball-img3.jpg'
OUTPUT_HEATMAP_PATH = 'lrp_overlay.png'

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
    for module in model.modules():
        def save_input_output(mod, inp, out):
            mod.input = inp[0]
            mod.output = out
        module.register_forward_hook(save_input_output)

def visualize_and_save_lrp(orig_pil: Image.Image,
                           attribution_tensor: torch.Tensor,
                           out_path: str = OUTPUT_HEATMAP_PATH):
    """
    Create a heatmap overlay of LRP attributions onto the original image,
    then save to 'out_path'.
    """
    # attribution_tensor: (1, 3, 224, 224)
    attr = attribution_tensor.squeeze(0).cpu().detach().numpy()  # → (3, 224, 224)
    #pos = np.clip(attr, a_min=0, a_max=None)
    #heatmap = pos.sum(axis=0) ## only pos contributions for visualization
    heatmap = attr.sum(axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()  # normalize to [0, 1]

    # Resize original PIL to 224×224 if needed
    orig_resized = orig_pil.resize((224, 224))
    orig_array = np.array(orig_resized).astype(np.float32) / 255.0

    # Plot and save (no plt.show)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(orig_array)
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
    ax.axis('off')
    plt.title("LRP Heatmap Overlay (target = basketball)")

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"LRP heatmap overlay saved to '{out_path}'.")


if __name__ == "__main__":
    print("cuda: ", )
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
    input_tensor, orig_pil = load_and_preprocess(IMAGE_FILEPATH, device=device)

    with torch.no_grad():
        output = augmentedVGG16(input_tensor)
        R = torch.zeros_like(output)
        R[0, TARGET_CLASS] = output[0, TARGET_CLASS]

    ### 4. Compute LRP attributions for the fixed TARGET_CLASS
    # Flatten model into an ordered list
    modules = list(augmentedVGG16.before) + [augmentedVGG16.encode, augmentedVGG16.decode] + list(augmentedVGG16.after) + list(augmentedVGG16.classifier)

    # Propagate in reverse order
    for module in reversed(modules):
        R = lrp(module, R, lrp_var='alphabeta', param=1.0)  # alpha=1, beta=0 

    ### 5. Visualize & save the heatmap overlay to disk
    visualize_and_save_lrp(orig_pil, R, out_path=OUTPUT_HEATMAP_PATH)
