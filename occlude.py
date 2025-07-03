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

        self.before = nn.Sequential(OrderedDict([
            ("conv1_1", nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ("relu1_1", nn.ReLU(inplace=True)),
            ("conv1_2", nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            ("relu1_2", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),

            ("conv2_1", nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ("relu2_1", nn.ReLU(inplace=True)),
            ("conv2_2", nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            ("relu2_2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),

            ("conv3_1", nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ("relu3_1", nn.ReLU(inplace=True)),
            ("conv3_2", nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ("relu3_2", nn.ReLU(inplace=True)),
            ("conv3_3", nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ("relu3_3", nn.ReLU(inplace=True)),
            ("pool3", nn.MaxPool2d(kernel_size=2, stride=2)),

            ("conv4_1", nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            ("relu4_1", nn.ReLU(inplace=True)),
            ("conv4_2", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ("relu4_2", nn.ReLU(inplace=True)),
            ("conv4_3", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ("relu4_3", nn.ReLU(inplace=True)),
            ("pool4", nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.after = nn.Sequential(OrderedDict([
            ("conv5_1", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ("relu5_1", nn.ReLU(inplace=True)),
            ("conv5_2", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ("relu5_2", nn.ReLU(inplace=True)),
            ("conv5_3", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ("relu5_3", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        # Dynamically compute flattened feature size
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_features = self.after(self.before(dummy_input))
            flattened_dim = dummy_features.view(1, -1).shape[1]

        self.classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(flattened_dim, 4096)),
            ("relu_fc1", nn.ReLU(inplace=True)),
            ("drop_fc1", nn.Dropout()),
            ("fc2", nn.Linear(4096, 4096)),
            ("relu_fc2", nn.ReLU(inplace=True)),
            ("drop_fc2", nn.Dropout()),
            ("fc3", nn.Linear(4096, 1000)),
        ]))

        self._load_weights(state_dict)

    def _load_weights(self, state_dict):
        self._load_submodule(self.before, state_dict, 'before')
        self._load_submodule(self.after, state_dict, 'after')
        self._load_submodule(self.classifier, state_dict, 'classifier')

    def _load_submodule(self, module, state_dict, prefix):
        for name, layer in module.named_children():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                w_key = f"{prefix}.{self._get_index(prefix, name)}.weight"
                b_key = f"{prefix}.{self._get_index(prefix, name)}.bias"
                if w_key in state_dict:
                    layer.weight.data.copy_(state_dict[w_key])
                    layer.bias.data.copy_(state_dict[b_key])

    def _get_index(self, prefix, layer_name):
        idx_map = {
            'before': {
                'conv1_1': 37, 'relu1_1': 36, 'conv1_2': 35, 'relu1_2': 34, 'pool1': 33,
                'conv2_1': 32, 'relu2_1': 31, 'conv2_2': 30, 'relu2_2': 29, 'pool2': 28,
                'conv3_1': 27, 'relu3_1': 26, 'conv3_2': 25, 'relu3_2': 24,
                'conv3_3': 23, 'relu3_3': 22, 'pool3': 21,
                'conv4_1': 20, 'relu4_1': 19, 'conv4_2': 18, 'relu4_2': 17,
                'conv4_3': 16, 'relu4_3': 15, 'pool4': 14,
            },
            'after': {
                'conv5_1': 13, 'relu5_1': 12, 'conv5_2': 11, 'relu5_2': 10,
                'conv5_3': 9,  'relu5_3': 8,  'pool5': 7,
            },
            'classifier': {
                'fc3': 0, 'drop_fc2': 1, 'relu_fc2': 2, 'fc2': 3,
                'drop_fc1': 4, 'relu_fc1': 5, 'fc1': 6,
            },
        }
        return idx_map[prefix][layer_name]

    def forward(self, x):
        x = self.before(x)
        x = self.after(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Load checkpoint
checkpoint = torch.load(PRUNED_CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint['state_dict']

model = PrunedVGG(state_dict).to(device)
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

