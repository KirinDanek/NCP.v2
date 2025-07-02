import argparse
import torch
from torchvision import models
import torch.nn as nn
from AugmentedVGG16 import *

USE_AUGMENTED_MODEL=True
SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = [3]  # test: ablate "ball" subspace (ix 3). Should be easy to see in LRP heatmap
U_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt'



def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='basketball_imagenet')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', default=True)
    
    # pruning config
    parser.add_argument('--relevance', action='store_true', default=True)
    parser.add_argument('--method_type', type=str, default='lrp')
    parser.add_argument('--pr_step', type=float, default=0.05)      # prune % per iteration
    parser.add_argument('--total_pr', type=float, default=0.8)     # prune % total

    args = parser.parse_args([])
    return args

def test_pruning_pipeline():
    args = get_test_args()

    if USE_AUGMENTED_MODEL:
        U = torch.load(U_FILEPATH)  # shape: (512, 512)
        U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)
        model = AugmentedVGG16(U_ab, U_ab_T)
    else:
        model = models.vgg16(pretrained=True)
        
    model.classifier[6] = nn.Linear(4096, 2) # for binary classification
    if args.cuda:
        model = model.cuda()

    print("Initializing PruningFineTuner...")
    if USE_AUGMENTED_MODEL:
        from prune_aug_vgg import PruningFineTuner
    else:
        from prune_van_vgg import PruningFineTuner
        
    tuner = PruningFineTuner(args, model)

    print("Training new 2-d output layer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # only train output layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[6].parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(15):  
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tuner.train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} complete. Avg Loss: {running_loss / len(tuner.train_loader):.4f}")


    print("Pruning...")
    tuner.prune()

    ### Save model weights and mid-pruning metrics
    #note: if augmented, augmented layers are removed prior to final fine tuning
    # Collect pruned structure info
    pruned_structure = [m.out_channels for m in tuner.model.features if isinstance(m, torch.nn.Conv2d)]

    # Save model + pruner metadata
    torch.save({
        'state_dict': tuner.model.state_dict(),
        'pruned_structure': pruned_structure,
        'train_loss': tuner.train_loss_tot,
        'test_loss': tuner.test_loss_tot,
        'test_acc': tuner.test_acc_tot,
        'test_iter': tuner.test_iter,
    }, "pruned_checkpoint.pth")


if __name__ == "__main__":
    test_pruning_pipeline()

