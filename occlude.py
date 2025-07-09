import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob 
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
import os

from collections import OrderedDict
from captum.attr import Occlusion


### vars
MODEL_VER = "ncp" ## ncp or van
MODEL_FILEPATH = f'/u/kd9132/n/fs/ncp/NCP.v2/results/pruned-models/{MODEL_VER}-basketball-80.pth'
IMAGE_DIR_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images/'
OUTPUT_HEATMAP_PATH = f'/u/kd9132/n/fs/ncp/NCP.v2/results/pruned-models/80-basketball-ablated/occlusion/{MODEL_VER}.png'
STRIDE = 1

# "basketball" in IN bball binary dataset
TARGET_CLASS = 0

# === De-normalize input for display ===
def denormalize(tensor, mean, std):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
# Load checkpoint
checkpoint = torch.load(MODEL_FILEPATH, map_location=device)
model = checkpoint['model'].to(device)
model.eval()

print("model initialized successfully")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device)
baseline = (imagenet_mean / imagenet_std).view(1, 3, 1, 1)

pattern = re.compile(r'img-(\d+)\.jpg')
image_paths = glob.glob(os.path.join(IMAGE_DIR_FILEPATH, 'img-*.jpg'))
for image_path in image_paths:

    # Load and preprocess input
    image = Image.open(image_path).convert("RGB")  # Replace with your image

    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    # Run for predicted class
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()
        print(f"Predicted label: {pred_label}, Raw output: {output.cpu().numpy()}")

    occlusion = Occlusion(model)
    attributions = occlusion.attribute(
        input_tensor,
        strides=(1, STRIDE, STRIDE),
        sliding_window_shapes=(1, 15, 15),
        target=TARGET_CLASS,
        baselines=baseline
    )
    attr = attributions.squeeze(0).cpu().detach().numpy()  # (3,224,224)
    heatmap = attr.sum(axis=0)  # debug can change to np.abs inb                        # (224,224)
    # Seismic-style visualization
    heatmap_seismic = 10 * ((np.abs(heatmap)**3.0).mean() ** (1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    # Get de-normalized image
    input_denorm = denormalize(input_tensor.squeeze(0).cpu(), imagenet_mean.cpu(), imagenet_std.cpu()).permute(1, 2, 0).numpy()

    # === Plot overlay ===
    fig, ax = plt.subplots()
    ax.imshow(input_denorm, interpolation='bilinear')
    ax.imshow(heatmap, cmap=my_cmap, alpha=0.5, vmin=-heatmap_seismic, vmax=heatmap_seismic, interpolation='bilinear')
    ax.axis('off')

    # Annotate predicted label and raw logits
    raw_logit = output[0, pred_label].item()
    ax.text(5, 20, f"Pred: {pred_label}, logit (sum(R)): {raw_logit:.2f}",
            fontsize=10, color='white', backgroundcolor='black')

    # Save heatmap overlay with label
    filename = os.path.basename(image_path)
    match = pattern.match(filename)
    if match is None:
        print(f"couldn't extract image idx from {filename}")
        continue
    img_idx = match.group(1)
    rule_output_path = OUTPUT_HEATMAP_PATH.replace('.png', f'_img-{img_idx}.png')
    os.makedirs(os.path.dirname(rule_output_path), exist_ok=True)

    plt.savefig(rule_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


