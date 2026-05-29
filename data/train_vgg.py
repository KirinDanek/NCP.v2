import os
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

from torchvision import transforms, models
from torchvision.datasets import ImageFolder


def get_dataloaders_with_holdout(
    image_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction_within_train: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Build train/val/test DataLoaders from an ImageFolder dataset with structure:

        image_root/
            blond_hair/
            not_blond/

    - blond_hair -> class index 0
    - not_blond -> class index 1

    Splitting:
        - test = test_fraction of full data (held out for *post-pruning* evaluation)
        - remaining 1 - test_fraction is split into train / val
          with val_fraction_within_train of that remainder used for val.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = ImageFolder(root=image_root, transform=transform)
    print("Found classes:", full_dataset.classes)
    print("class_to_idx:", full_dataset.class_to_idx)

    expected_mapping = {"blond_hair": 0, "not_blond": 1}
    if full_dataset.class_to_idx != expected_mapping:
        raise RuntimeError(
            f"Expected class_to_idx {expected_mapping}, "
            f"but got {full_dataset.class_to_idx}. "
            "Check your folder names / ordering."
        )

    n_total = len(full_dataset)
    n_test = int(test_fraction * n_total)
    n_trainval = n_total - n_test

    # Reproducible split
    g = torch.Generator().manual_seed(seed)
    trainval_dataset, test_dataset = random_split(full_dataset, [n_trainval, n_test], generator=g)

    # Now split trainval into train and val
    n_val = int(val_fraction_within_train * n_trainval)
    n_train = n_trainval - n_val
    train_dataset, val_dataset = random_split(trainval_dataset, [n_train, n_val], generator=g)

    print(f"Total: {n_total} | Train: {n_train} | Val: {n_val} | Test (held-out): {n_test}")
    print("Train class counts:", torch.bincount(torch.tensor([full_dataset.targets[i] for i in train_dataset.indices])))
    print("Val class counts:", torch.bincount(torch.tensor([full_dataset.targets[i] for i in val_dataset.indices])))
    print("Test class counts:", torch.bincount(torch.tensor([full_dataset.targets[i] for i in test_dataset.indices])))



    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_dataset.class_to_idx


def train_vgg16_blond_notblond(
    image_root: str,
    model_path: str,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 4,
    val_fraction_within_train: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
):
    """
    Train VGG16 with 2 logits [blond, not-blond] using the ImageFolder structure,
    while holding out test_fraction of data for *post-pruning* testing.

    Returns:
        model (on CPU with best weights),
        class_to_idx mapping,
        test_loader (for later evaluation)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders_with_holdout(
        image_root=image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction_within_train=val_fraction_within_train,
        test_fraction=test_fraction,
        seed=seed,
    )

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)  # [logit_blond, logit_not_blond]
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)  # [B, 2]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Val ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    if best_state_dict is None:
        best_state_dict = model.state_dict()

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(best_state_dict, model_path)
    print(f"Saved best model (best val_acc = {best_val_acc:.4f}) to {model_path}")

    # Load best weights on CPU for later XAI / pruning
    model.cpu()
    model.load_state_dict(best_state_dict)

    return model, class_to_idx, test_loader


if __name__ == "__main__":
    image_root = "/n/fs/ncp/NCP.v2/data/images/celeba_hair_color"
    model_path = "/n/fs/ncp/NCP.v2/data/trained_models/vgg16_celeba_blond2.pth"

    model, class_to_idx, test_loader = train_vgg16_blond_notblond(
        image_root=image_root,
        model_path=model_path,
        num_epochs=8,
        batch_size=64,
        lr=1e-4,
        num_workers=4,
        val_fraction_within_train=0.1,
        test_fraction=0.2,
        seed=42,
    )
    print("Final class_to_idx:", class_to_idx)
    print("Held-out test batches:", len(test_loader))
