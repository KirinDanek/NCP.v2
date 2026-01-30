import torch
import data as dataset
from torchvision import datasets, transforms, models
from AugmentedVGG16 import AugmentedVGG16
#from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

MODEL_PATH = '/n/fs/ncp/NCP.v2/results/pruned-models/80-carton-dugong-orig-wm-abl/van.pth'
DEVICE = 'cuda'

print('testing ', MODEL_PATH)
train_dataset, test_dataset = dataset.get_carton_imagenet()

print(f"train_dataset:{len(train_dataset)}, test_dataset:{len(test_dataset)}")
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=32,
                                            shuffle=False)

train_num = len(train_loader)
test_num = len(test_loader)

model = torch.load(MODEL_PATH)['model']
model = model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ----- 5. Compute metrics -----
accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item() * 100
print(f"Accuracy:  {accuracy:.2f}%")
'''
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Accuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ----- 6. Optional detailed breakdown -----
print("\nDetailed classification report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
'''