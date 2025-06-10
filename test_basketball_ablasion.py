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
    Save a standalone LRP heatmap image (no overlay or title).
    """
    attr = attribution_tensor.squeeze(0).cpu().detach().numpy()  # → (3, 224, 224)

    if attr.shape != (3, 224, 224):
        raise ValueError(f"Expected attribution_tensor of shape (1, 3, 224, 224), got {attr.shape}")

    heatmap = attr.sum(axis=0)  # → (224, 224)
    heatmap = np.maximum(heatmap, 0)

    max_val = heatmap.max()
    if max_val == 0:
        raise ValueError("Heatmap is all zeros, cannot normalize.")
    if not np.isfinite(heatmap).all():
        raise ValueError("Heatmap contains NaNs or infs.")

    heatmap /= max_val

    arr = R.squeeze(0).cpu().detach().numpy()
    print("Relevance stats — min:", arr.min(), "max:", arr.max(), "mean:", arr.mean(), "nonzero count:", np.count_nonzero(arr))

    plt.imsave(out_path, heatmap, cmap='hot')
    print(f"LRP heatmap saved to '{out_path}'.")



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
    input_tensor, _ = load_and_preprocess(IMAGE_FILEPATH, device=device)

    with torch.no_grad():
        output = augmentedVGG16(input_tensor)
        R = torch.zeros_like(output)
        R[0, TARGET_CLASS] = output[0, TARGET_CLASS]

    #debug
    assert any(hasattr(m, "input") for m in augmentedVGG16.modules()), "Forward hook registration failed"
    print("Encode weight NaNs:", torch.isnan(augmentedVGG16.encode.weight).any().item())
    print("Decode weight NaNs:", torch.isnan(augmentedVGG16.decode.weight).any().item())


    print("output relevance: ", R.sum())
    ### 4. Compute LRP attributions for the fixed TARGET_CLASS
    # Flatten model into an ordered list
    modules = list(augmentedVGG16.before) + [augmentedVGG16.encode, augmentedVGG16.decode] + list(augmentedVGG16.after) + list(augmentedVGG16.classifier)

    # Propagate in reverse order
    for module in reversed(modules):
        R = lrp(module, R, lrp_var='alphabeta', param=1.0)  # alpha=1, beta=0 
        print(f"After {module.__class__.__name__}, relevance sum: {R.sum().item()}, has NaN: {torch.isnan(R).any().item()}")
    print("input relevance: ", R.sum())
    ### 5. Visualize & save the heatmap overlay to disk
    visualize_and_save_lrp(R, out_path=OUTPUT_HEATMAP_PATH)
