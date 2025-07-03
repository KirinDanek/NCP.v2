import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
import os

from collections import OrderedDict

PRUNED_CHECKPOINT_PATH = "/u/kd9132/n/fs/ncp/NCP.v2/results/pruned-models/aug-basketball-80.pth"
IMAGE_PATH = "/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images/img-3.jpg"
OUT_PATH = "./ball_pruned_heatmap.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrunedVGG(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.features = self._build_feature_layers(state_dict)
        self.classifier = self._build_classifier_layers(state_dict)

    def _build_feature_layers(self, state_dict):
        layers = []
        i = 0
        while f'features.{i}.weight' in state_dict:
            weight = state_dict[f'features.{i}.weight']
            bias = state_dict[f'features.{i}.bias']
            if len(weight.shape) == 4:  # Conv2d
                conv = nn.Conv2d(
                    in_channels=weight.shape[1],
                    out_channels=weight.shape[0],
                    kernel_size=weight.shape[2],
                    padding=1  # assuming same padding
                )
                conv.weight = nn.Parameter(weight)
                conv.bias = nn.Parameter(bias)
                layers.append(conv)
            elif 'ReLU' in f'features.{i}':
                layers.append(nn.ReLU(inplace=True))
            elif 'MaxPool' in f'features.{i}':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            i += 1
        return nn.Sequential(*layers)

    def _build_classifier_layers(self, state_dict):
        layers = []
        i = 0
        while f'classifier.{i}.weight' in state_dict:
            weight = state_dict[f'classifier.{i}.weight']
            bias = state_dict[f'classifier.{i}.bias']
            if len(weight.shape) == 2:  # Linear
                fc = nn.Linear(
                    in_features=weight.shape[1],
                    out_features=weight.shape[0]
                )
                fc.weight = nn.Parameter(weight)
                fc.bias = nn.Parameter(bias)
                layers.append(fc)
            elif 'ReLU' in f'classifier.{i}':
                layers.append(nn.ReLU(inplace=True))
            elif 'Dropout' in f'classifier.{i}':
                layers.append(nn.Dropout(p=0.5))
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Load checkpoint
checkpoint = torch.load(PRUNED_CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint['state_dict']

model = PrunedVGG(state_dict)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("model initialized successfully")

print("loading weights")

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
    print(f"Predicted label: {pred_label}, Raw output: {output.cpu().numpy()}")
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

