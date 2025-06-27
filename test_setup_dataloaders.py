import torch
from prune_van_vgg import PruningFineTuner
from types import SimpleNamespace
import torch.nn as nn
from torch.nn import Sequential as Seq

class MiniVGG(nn.Module):
    def __init__(self, num_classes=2):
        super(MiniVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def test_setup_dataloaders():
    args = SimpleNamespace(
        seed=42,
        cuda=torch.cuda.is_available(),
        data_type='basketball_imagenet',
        train_batch_size=8,
        test_batch_size=4
    )

    # Dummy model to pass into the constructor
    model = MiniVGG(num_classes=2)

    print("Initializing PruningFineTuner...")
    trainer = PruningFineTuner(args, model)

    print("\nIterating over training data:")
    for batch_idx, (data, target) in enumerate(trainer.train_loader):
        print(f"Batch {batch_idx} - data shape: {data.shape}, labels: {target.tolist()}")
        if batch_idx == 1:
            break

    print("\nIterating over test data:")
    for batch_idx, (data, target) in enumerate(trainer.test_loader):
        print(f"Batch {batch_idx} - data shape: {data.shape}, labels: {target.tolist()}")
        if batch_idx == 1:
            break

if __name__ == "__main__":
    test_setup_dataloaders()
