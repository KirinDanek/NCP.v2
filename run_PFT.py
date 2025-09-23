import argparse
import torch
torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0)) ## debug
from torchvision import models
import torch.nn as nn
from AugmentedVGG16 import *
import os

USE_AUGMENTED_MODEL=True
FINE_TUNE_CONV_LAYERS = True # False: classifier only


SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = [1]  # crate wm subspace is idx 1
U_FILEPATH = '/n/fs/ncp/NCP.v2/data/projection_matrices/U_crate_tensor.pt'
OUT_DIR = '/n/fs/ncp/NCP.v2/results/pruned-models/80-crate-0p5_wm-packet-wm_abl/'

# if using already-pruned model
#MODEL_VER = 'ncp'
#MODEL_FILEPATH = f'/u/kd9132/n/fs/ncp/NCP.v2/results/pruned-models/{MODEL_VER}-crate-80.pth'

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='crate_imagenet')
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
    parser.add_argument('--total_pr', type=float, default=0.80)     # prune % total

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
    
    '''
    ### load already pruned model to continue pruning
    checkpoint = torch.load(MODEL_FILEPATH)
    model = checkpoint['model']
    model.augmented = True
    
    U = torch.load(U_FILEPATH)  # shape: (512, 512)
    U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)
    # Build 1x1 convs for encoding (UT) and decoding (U)
    model.encode = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False) #1x1 conv layer --> linear layer, channelwise
    model.decode = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False)
    
    # Copy UT into encode.weight and U into decode.weight
    # UT and U should be torch.Tensors of shape (512,512)
    with torch.no_grad():
        model.encode.weight.copy_(U_ab_T.view(512, 512, 1, 1)) 
        model.decode.weight.copy_(U_ab.view(512, 512, 1, 1))
    
    # Freeze encode/decode weights
    for param in model.encode.parameters():
        param.requires_grad = False
    for param in model.decode.parameters():
        param.requires_grad = False
    '''
    if args.cuda:
        print('cuda status: ', torch.cuda.is_available())
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
    for epoch in range(15): #15  
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tuner.train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad() # clear gradients 
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} complete. Avg Loss: {running_loss / len(tuner.train_loader):.4f}")
    

    print("Pruning...")
    tuner.prune(fine_tune_without_augmented_layers=True)
    
    ### Save model weights and mid-pruning metrics
    #note: if augmented, augmented layers are removed prior to final fine tuning

    if USE_AUGMENTED_MODEL:
        pruned_structure = []
        for module in tuner.model.before:
            if isinstance(module, torch.nn.Conv2d):
                pruned_structure.append(module.out_channels)
        for module in tuner.model.after:
            if isinstance(module, torch.nn.Conv2d):
                pruned_structure.append(module.out_channels)
    else:    
        # Collect pruned structure info
        pruned_structure = [m.out_channels for m in tuner.model.features if isinstance(m, torch.nn.Conv2d)]
    if USE_AUGMENTED_MODEL:
        out_path = os.path.join(OUT_DIR, 'ncp_ft_without_aug.pth')
    else:
        out_path = os.path.join(OUT_DIR, 'van.pth')
    os.makedirs(OUT_DIR, exist_ok=True)
    # Save model + pruner metadata
    torch.save({
        'model': tuner.model,
        'state_dict': tuner.model.state_dict(),
        'pruned_structure': pruned_structure,
        'train_loss': tuner.train_loss_tot,
        'test_loss': tuner.test_loss_tot,
        'test_acc': tuner.test_acc_tot,
        'test_iter': tuner.test_iter,
    }, out_path)
    

if __name__ == "__main__":
    test_pruning_pipeline()

