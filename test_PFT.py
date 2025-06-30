import argparse
import torch
from torchvision import models
import torch.nn as nn
from prune_van_vgg import PruningFineTuner

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
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 2) # for binary classification
    if args.cuda:
        model = model.cuda()

    print("Initializing PruningFineTuner...")
    tuner = PruningFineTuner(args, model)

    print("Training new 2-d output layer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # only train output layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    tuner.train_epoch(optimizer=optimizer)

    print("Getting pruning candidates...")
    candidates = tuner.get_candidates_to_prune(num_filters_to_prune=10)
    print("Top 10 candidates to prune:", candidates)

    print("Running full pruning pipeline (short version)...")
    tuner.prune()

    ### TODO: save model weights and mid-pruning metrics
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

