import torch
import numpy as np
import NCP
from NCP import AugmentedVGG16, ablate_subspace_matrix

from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt
from captum.attr import LRP

### vars
SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = [3]  # test: ablate "ball" subspace. Should be easy to see in LRP heatmap
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


def visualize_and_save_lrp(orig_pil: Image.Image,
                           attribution_tensor: torch.Tensor,
                           out_path: str = OUTPUT_HEATMAP_PATH):
    """
    Create a heatmap overlay of LRP attributions onto the original image,
    then save to 'out_path'.
    """
    # attribution_tensor: (1, 3, 224, 224)
    attr = attribution_tensor.squeeze(0).cpu().detach().numpy()  # → (3, 224, 224)
    pos = np.clip(attr, a_min=0, a_max=None)
    heatmap = pos.sum(axis=0) ## only pos contributions for visualization

    heatmap -= heatmap.min()
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    ### 1. Load tensor U, ablate and move to GPU
    U = torch.load(U_FILEPATH)  # shape: (512, 512)
    U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)
    U_ab = U_ab.to(device)
    U_ab_T = U_ab_T.to(device)

    ### 2. Build the augmented model, send to GPU, set eval()
    augmentedVGG16 = AugmentedVGG16(U=U_ab, UT=U_ab_T).to(device)
    augmentedVGG16.eval()

    ### 3. Load and preprocess the input image
    input_tensor, orig_pil = load_and_preprocess(IMAGE_FILEPATH, device=device)

    ### 4. Compute LRP attributions for the fixed TARGET_CLASS
    input_tensor.requires_grad_(True)
    lrp = LRP(augmentedVGG16, rule_type='alpha-beta', alpha=1, beta=0)  
    attributions = lrp.attribute(input_tensor, target=TARGET_CLASS)  # → (1, 3, 224, 224)

    ### 5. Visualize & save the heatmap overlay to disk
    visualize_and_save_lrp(orig_pil, attributions, out_path=OUTPUT_HEATMAP_PATH)
