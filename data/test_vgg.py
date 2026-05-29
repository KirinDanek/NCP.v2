import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


# ---------- 1. Dataloaders with the SAME split logic as training ----------

def get_dataloaders_with_holdout(
    image_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction_within_train: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
):
    """
    Build train/val/test loaders, holding out test_fraction of data for later.
    MUST match the settings used during training to get the same val set.
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

    g = torch.Generator().manual_seed(seed)
    trainval_dataset, test_dataset = random_split(
        full_dataset, [n_trainval, n_test], generator=g
    )

    n_val = int(val_fraction_within_train * n_trainval)
    n_train = n_trainval - n_val
    train_dataset, val_dataset = random_split(
        trainval_dataset, [n_train, n_val], generator=g
    )

    print(f"Total: {n_total} | Train: {n_train} | Val: {n_val} | Test (held-out): {n_test}")

    # Optional class counts (sanity check)
    print(
        "Train class counts:",
        torch.bincount(torch.tensor([full_dataset.targets[i] for i in train_dataset.indices])),
    )
    print(
        "Val class counts:",
        torch.bincount(torch.tensor([full_dataset.targets[i] for i in val_dataset.indices])),
    )
    print(
        "Test class counts:",
        torch.bincount(torch.tensor([full_dataset.targets[i] for i in test_dataset.indices])),
    )

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


# ---------- 2. Metric computation on a loader ----------

def compute_per_class_metrics(model, dataloader, device, class_names=None):
    """
    Computes per-class accuracy, precision, recall and overall accuracy
    for a 2-class classifier (blond_hair=0, not_blond=1).
    """
    model.eval()
    num_classes = 2
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confmat[int(t), int(p)] += 1

    confmat = confmat.float()
    tp = confmat.diag()
    support = confmat.sum(dim=1)    # true counts per class
    predicted = confmat.sum(dim=0)  # predicted counts per class

    per_class_acc = tp / support.clamp(min=1)
    recall = tp / support.clamp(min=1)
    precision = tp / predicted.clamp(min=1)
    overall_acc = tp.sum() / confmat.sum().clamp(min=1)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    print("\n=== Validation metrics ===")
    print("Overall val accuracy:", overall_acc.item())
    for i, name in enumerate(class_names):
        print(
            f"{name}: "
            f"acc={per_class_acc[i].item():.4f}, "
            f"precision={precision[i].item():.4f}, "
            f"recall={recall[i].item():.4f}"
        )
    print("Confusion matrix (rows=true, cols=pred):\n", confmat)

    return {
        "confusion_matrix": confmat.cpu(),
        "overall_accuracy": overall_acc.item(),
        "per_class_accuracy": per_class_acc.cpu().tolist(),
        "per_class_precision": precision.cpu().tolist(),
        "per_class_recall": recall.cpu().tolist(),
    }


# ---------- 3. Glue: load model from model_path and evaluate on val ----------

def eval_val_from_checkpoint(
    image_root: str,
    model_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction_within_train: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader, _, class_to_idx = get_dataloaders_with_holdout(
        image_root=image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction_within_train=val_fraction_within_train,
        test_fraction=test_fraction,
        seed=seed,
    )

    # Rebuild the same architecture: VGG16 with 2 logits
    model = models.vgg16(weights=None)  # weights=None avoids re-downloading ImageNet
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)  # [logit_blond, logit_not_blond]

    # Load checkpoint
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Class names in correct index order
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[0], idx_to_class[1]]

    metrics = compute_per_class_metrics(model, val_loader, device, class_names=class_names)
    return metrics


if __name__ == "__main__":
    image_root = "/n/fs/ncp/NCP.v2/data/images/celeba_hair_color"  # your ImageFolder root
    model_path = "/n/fs/ncp/NCP.v2/data/trained_models/vgg16_celeba_blond2.pth"  # your checkpoint

    eval_val_from_checkpoint(
        image_root=image_root,
        model_path=model_path,
        batch_size=64,
        num_workers=2,          # match what you used in training
        val_fraction_within_train=0.1,
        test_fraction=0.2,
        seed=42,                # must match training
    )
