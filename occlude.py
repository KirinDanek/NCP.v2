import torch
import torch.nn as nn
from AugmentedVGG16 import AugmentedVGG16
from torchvision import transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
import os

PRUNED_CHECKPOINT_PATH = "/u/kd9132/n/fs/ncp/NCP.v2/results/pruned-models/aug-basketball-80.pth"
IMAGE_PATH = "/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images/img-3.jpg"
OUT_PATH = "./ball_pruned_heatmap.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(PRUNED_CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint['state_dict']

# Rebuild model architecture
model = AugmentedVGG16(U=torch.eye(512), UT=torch.eye(512))  # Make sure this matches the saved model
model.augmented = False  # Augmented layers were removed

# If you modified classifier for binary classification (e.g. 2 classes)
model.classifier[6] = nn.Linear(4096, 2)

# Load weights
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Load and preprocess input
image = Image.open(IMAGE_PATH).convert("RGB")  # Replace with your image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]


def occlusion_heatmap(model, input_tensor, label_idx, occlusion_size=16, occlusion_stride=8):
    model.eval()
    input_tensor = input_tensor.clone()

    _, _, H, W = input_tensor.shape
    heatmap = np.zeros((H, W))
    base_output = model(input_tensor)
    base_score = base_output[0, label_idx].item()

    for y in range(0, H, occlusion_stride):
        for x in range(0, W, occlusion_stride):
            # Clone and occlude
            occluded = input_tensor.clone()
            occluded[:, :, y:y+occlusion_size, x:x+occlusion_size] = 0.0

            with torch.no_grad():
                output = model(occluded)
                score = output[0, label_idx].item()

            relevance = base_score - score
            heatmap[y:y+occlusion_size, x:x+occlusion_size] += relevance

    return heatmap

# Run for predicted class
with torch.no_grad():
    output = model(input_tensor)
    pred_label = output.argmax(dim=1).item()

heatmap = occlusion_heatmap(model, input_tensor, pred_label)

# Seismic-style visualization
heatmap_seismic = 10 * ((np.abs(heatmap)**3.0).mean() ** (1.0/3))
my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
my_cmap[:, 0:3] *= 0.85
my_cmap = ListedColormap(my_cmap)

plt.figure()
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.axis('off')
plt.imshow(heatmap, cmap=my_cmap, vmin=-heatmap_seismic, vmax=heatmap_seismic, interpolation='nearest')

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH)
plt.close()
print(f"Saved seismic heatmap to: {OUT_PATH}")

