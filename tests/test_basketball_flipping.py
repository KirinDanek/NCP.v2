import torch
import numpy as np
import AugmentedVGG16
from torch import nn
from AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix

from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt

from captum.attr import Occlusion

### vars
SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = [0,1,2]  # test: ablate "ball" subspace. Should be easy to see in LRP heatmap
U_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt'
IMAGE_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa-basketball-img3.jpg'
OUTPUT_HEATMAP_PATH = 'lrp_overlay.png'

# "basketball" in ImageNet is class index 430 (zero‐indexed)
TARGET_CLASS = 430


def load_and_preprocess(image_path: str, device: torch.device):
    """
    Load an image, crop it exactly to 224×224 for VGG16, then normalize.
    Returns (preprocessed_tensor, cropped_PIL_image).
    """
    # 1) Crop to 224×224 so we have a PIL that's the same size as our heatmap
    crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    img = Image.open(image_path).convert('RGB')
    img_cropped = crop(img)  # now exactly 224×224

    # 2) Convert to tensor + normalize
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = normalize(img_cropped).unsqueeze(0).to(device)  # (1,3,224,224)

    return tensor, img_cropped

def visualize_and_save_lrp(orig_pil: Image.Image,
                           attribution_tensor: torch.Tensor,
                           out_path: str = OUTPUT_HEATMAP_PATH):
    """
    Overlay a 224×224 heatmap onto a 224×224 PIL, disable interpolation/aspect changes,
    and save to disk.
    """
    # 1) Build the heatmap array
    attr = attribution_tensor.squeeze(0).cpu().detach().numpy()  # (3,224,224)
    heatmap = np.abs(attr).sum(axis=0)                          # (224,224)
    heatmap -= heatmap.min()
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    # 2) Convert the already‐cropped PIL (224×224) to array [0..1]
    orig_array = np.array(orig_pil).astype(np.float32) / 255.0  # shape (224,224,3)

    # 3) Plot with matching shapes, no interpolation, equal aspect
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(orig_array, interpolation='none', aspect='equal')
    ax.imshow(heatmap, cmap='jet', alpha=0.4,
              interpolation='none', aspect='equal',
              vmin=0, vmax=1)
    ax.axis('off')
    plt.title("LRP Heatmap Overlay")

    # 4) Save to file
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"LRP heatmap overlay saved to '{out_path}'.")

if __name__ == "__main__":
    ###DEBUG
    print("cuda: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406],
                             device=device).view(1, 3, 1, 1)

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

    ### DEBUG 
    print("Model parameters on:", next(augmentedVGG16.parameters()).device)
    print("Input tensor on:", input_tensor.device)
    ### 4. pixel flipping attributions
    occlusion = Occlusion(augmentedVGG16)
    attributions = occlusion.attribute(
        input_tensor,
        strides=(1, 8, 8),
        sliding_window_shapes=(1, 15, 15),
        target=TARGET_CLASS,
        baselines=imagenet_mean
    )
    
    ### DEBUG. Forward pass to pick top‐predicted class
    with torch.no_grad():
        logits = augmentedVGG16(input_tensor)     # → (1, 1000)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_catid = torch.max(probs, dim=1)
        target_class = top_catid.item()
        print(f"Target class = {target_class}  (prob={top_prob.item():.4f})")

    ### 5. Visualize & save the heatmap overlay to disk
    visualize_and_save_lrp(orig_pil, attributions, out_path=OUTPUT_HEATMAP_PATH)
