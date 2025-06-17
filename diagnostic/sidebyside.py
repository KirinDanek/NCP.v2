import torch
import numpy as np
import AugmentedVGG16
from torch import nn
from AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix
from lrp import *

from torchvision import transforms
from PIL import Image 
import matplotlib.pyplot as plt

from torchvision.models import vgg16


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
        return 0.0
    elif 7 <= module_idx <= 13:                 # Conv5
        return 0.0
    elif 14 <= module_idx <= 20:                # Conv4
        return 0.10
    elif 21 <= module_idx <= 27:                # Conv3
        return 0.25
    else:
        if module_idx < 28 or module_idx > 37:
            print(f'unexpected module index {module_idx}')                                       # Conv2, Conv1, and anything earlier
        return 0.50

if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### 1. Load tensor U, ablate and move to GPU
    U = torch.load(U_FILEPATH)  # shape: (512, 512)
    U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)
    U_ab = U_ab.to(device)
    U_ab_T = U_ab_T.to(device)

    ### 2. Build models
    augmentedVGG16 = AugmentedVGG16(U=U_ab, UT=U_ab_T).to(device).eval()
    vanillaVGG16 = vgg16(pretrained=True).to(device).eval()

    ### 3. Register LRP hooks
    register_hooks(augmentedVGG16)
    register_hooks(vanillaVGG16)

    ### 4. Run dummy forward pass to activate hooks if needed
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    _ = vanillaVGG16(dummy_input)
    _ = augmentedVGG16(dummy_input)

    ### 5. Compare hooks
    for i in range(14):
        module_vanilla = vanillaVGG16.features[i]
        module_augmented = augmentedVGG16.features[i]

        print(f"\n--- Layer {i}: {module_vanilla.__class__.__name__} ---")
        print("Vanilla forward hooks:", list(module_vanilla._forward_hooks.keys()))
        print("Augmented forward hooks:", list(module_augmented._forward_hooks.keys()))
        print("Vanilla backward hooks:", list(module_vanilla._backward_hooks.keys()))
        print("Augmented backward hooks:", list(module_augmented._backward_hooks.keys()))

        # Optional: Check hook function identity
        v_fhooks = list(module_vanilla._forward_hooks.values())
        a_fhooks = list(module_augmented._forward_hooks.values())
        print("Forward hook match:", v_fhooks == a_fhooks)


