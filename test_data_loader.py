import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from data import get_basketball_imagenet


def main():
    train_set, test_set = get_basketball_imagenet()
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

    from collections import Counter
    all_labels = [label for _, label in train_set]
    print(Counter(all_labels))
    # Show a few samples from the training set
    print("Showing a few training examples...")
    loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    print("shape: ", images.shape)
    print("Labels:", labels.tolist())
    #imshow(make_grid(images), title="Labels: " + ', '.join(str(l.item()) for l in labels))

if __name__ == "__main__":
    main()
