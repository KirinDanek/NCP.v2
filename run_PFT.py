"""
run_PFT.py

Purpose
-------
Entry-point script to run the pruning + fine-tuning (PFT) pipeline end-to-end.

Responsibilities
---------------
- Reads all experiment configuration from CLI arguments (--pruner, --u_filepath,
  --out_dir, --subspace_dims, --irrelevant_subspaces, --fine_tune_conv_layers, etc.)
- Builds model:
    * ncp:     loads U (.pt or .npy), applies ablation, instantiates AugmentedVGG16
    * vanilla: loads torchvision vgg16 (ImageNet-pretrained by default)
  Then replaces classifier[6] with a 2-way Linear for binary classification.
- Optionally loads a custom pretrained checkpoint (--pretrained_model_path) to
  override ImageNet weights.  Handles the fairness_harness checkpoint format
  {'state_dict': ..., 'val_metrics': ..., ...} as well as plain state_dicts.
  For the ncp pruner the vanilla VGG16 keys (features.*) are automatically
  remapped to AugmentedVGG16's split layout (before.* / after.*).
- Optionally trains the final classifier layer before pruning (warm start).
- Instantiates the appropriate VGGAdapter, then PruningFineTuner from prune_vgg.py,
  and calls prune().
- Saves pruned model + metadata (structure, loss/acc curves, iteration indices,
  per-class precision/recall/accuracy, per-subgroup fairness stats).

Important notes
---------------
- The "test" loader inside PruningFineTuner is treated as prune_val during pruning
  (not the final downstream evaluation set).
- For augmented models, encode/decode are frozen and deleted after pruning.
- Output saving includes both the full model object and state_dict; prefer state_dict
  for portability across code changes.

Reproducibility
---------------
- Ensure args.seed is set and passed consistently to data splits and DataLoader shuffling.
- Keep a record of celeba split fractions/seed to recreate downstream_test later.

Example invocations
-------------------
# Vanilla pruning from CelebA-finetuned checkpoint:
python run_PFT.py --pruner vanilla --data_type celeba_lipstick \\
    --pretrained_model_path /n/fs/ncp/drsa-demo/fairness_harness_outputs/\\
Wearing_Lipstick/seed0/checkpoints/best_model.pth \\
    --out_dir /n/fs/ncp/NCP.v2/results/pruned-models/celeba-lipstick-vanilla/

# NCP pruning from same checkpoint, inserting concept layer from U_sx16:
python run_PFT.py --pruner ncp --data_type celeba_lipstick \\
    --pretrained_model_path /n/fs/ncp/drsa-demo/fairness_harness_outputs/\\
Wearing_Lipstick/seed0/checkpoints/best_model.pth \\
    --u_filepath /n/fs/ncp/drsa-demo/data/projection_matrices/celeba/\\
Wearing_Lipstick/seed0/conv4_3/U_sx16.npy \\
    --subspace_dims $(python -c "print(' '.join(['32']*16))") \\
    --irrelevant_subspaces 0 \\
    --out_dir /n/fs/ncp/NCP.v2/results/pruned-models/celeba-lipstick-ncp/
"""


import argparse
import numpy as np
import torch
torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0)) ## debug
from torchvision import models
import torch.nn as nn
from AugmentedVGG16 import *
import os


# ---------------------------------------------------------------------------
# Checkpoint / U-matrix loading helpers
# ---------------------------------------------------------------------------

def _load_u_matrix(path: str) -> torch.Tensor:
    """Load the projection matrix U from a .pt or .npy file.

    Returns a 2-D (d_in, d_out) float32 tensor suitable for ablate_subspace_matrix.

    If the file contains a 3-D array of shape (d, n_subspaces, subspace_dim)
    — as produced by U_sx*.npy files from the drsa-demo pipeline — it is
    automatically reshaped to (d, n_subspaces * subspace_dim).
    Set --subspace_dims accordingly: e.g. for shape (512, 16, 32) use
    --subspace_dims 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32
    """
    if path.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 3:
            d, n_sub, sub_dim = arr.shape
            print(f"[U matrix] 3-D array {arr.shape} → reshaping to ({d}, {n_sub * sub_dim})")
            arr = arr.reshape(d, n_sub * sub_dim)
        return torch.tensor(arr, dtype=torch.float32)
    else:
        U = torch.load(path, map_location='cpu')
        if not isinstance(U, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor from {path!r}, got {type(U)}")
        return U.float()


def _load_checkpoint_state_dict(path: str) -> dict:
    """Extract a state_dict from a checkpoint file.

    Handles checkpoints saved as:
      - Plain OrderedDict of parameter tensors (state_dict itself)
      - {'state_dict': ..., ...}   — fairness_harness / common trainer format
      - {'model': <nn.Module>, ...} — full model object saved in dict
    """
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        if 'model' in ckpt:
            obj = ckpt['model']
            return obj.state_dict() if hasattr(obj, 'state_dict') else obj
        # Assume the dict itself is the state_dict
        return ckpt
    if hasattr(ckpt, 'state_dict'):
        return ckpt.state_dict()
    raise ValueError(f"Cannot extract state_dict from checkpoint at {path!r}")


def _remap_vgg16_features_to_augmented(state_dict: dict) -> dict:
    """Remap vanilla VGG16 state_dict keys to AugmentedVGG16's attribute layout.

    AugmentedVGG16 splits VGG16's features sequential at index 23:
      before  = features[0..22]   (Conv1_1 through ReLU after conv4_3)
      encode  = 1×1 conv set from U  — NOT loaded from checkpoint
      decode  = 1×1 conv set from U  — NOT loaded from checkpoint
      after   = features[23..30]  (pool4 through pool5)

    Remapping applied:
      features.{i}.*  →  before.{i}.*        for i ∈ 0..22
      features.{i}.*  →  after.{i - 23}.*   for i ∈ 23..30
      classifier.*    →  classifier.*         (unchanged)
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            parts = k.split('.')
            layer_idx = int(parts[1])
            suffix = '.'.join(parts[2:])
            if layer_idx < 23:
                new_sd[f'before.{layer_idx}.{suffix}'] = v
            else:
                new_sd[f'after.{layer_idx - 23}.{suffix}'] = v
        else:
            new_sd[k] = v
    return new_sd


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='celeba_lipstick')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')

    # pruning config
    parser.add_argument('--relevance', action='store_true', default=True)
    parser.add_argument('--no_relevance', dest='relevance', action='store_false')
    parser.add_argument('--method_type', type=str, default='lrp')
    parser.add_argument('--pr_step', type=float, default=0.05)      # prune % per iteration
    parser.add_argument('--total_pr', type=float, default=0.80)     # prune % total
    parser.add_argument('--rank_n', type=int, default=10000,
                    help='Number of training samples to use for ranking filters per pruning iteration')
    parser.add_argument('--min_male_with_attr', type=int, default=200,
                        help='Min Male+attr samples in eval set; supplement from the official '
                             'TEST split if needed. 0=disabled. Typical value for Wearing_Lipstick: 200.')

    # model / experiment config
    parser.add_argument('--pruner', type=str, choices=['ncp', 'vanilla'], default='ncp',
                        help='Use concept-augmented NCP pruner or vanilla VGG16 pruner')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='Path to a pretrained checkpoint to use instead of ImageNet weights. '
                             'Supports fairness_harness format {state_dict, val_metrics, ...} and '
                             'plain state_dicts. For --pruner ncp, vanilla VGG16 keys (features.*) '
                             'are automatically remapped to AugmentedVGG16 layout.')
    parser.add_argument('--fine_tune_conv_layers', action='store_true', default=True)
    parser.add_argument('--no_fine_tune_conv_layers', dest='fine_tune_conv_layers',
                        action='store_false')
    parser.add_argument('--fine_tune_without_augmented_layers', action='store_true', default=False)
    parser.add_argument('--subspace_dims', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--irrelevant_subspaces', type=int, nargs='+', default=[],
                        help='Indices of concept subspaces to ablate (0-indexed)')
    parser.add_argument('--u_filepath', type=str,
                        default='/n/fs/ncp/NCP.v2/data/projection_matrices/')
    parser.add_argument('--out_dir', type=str,
                        default='/n/fs/ncp/NCP.v2/results/')

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def test_pruning_pipeline():
    args = get_test_args()

    # ------------------------------------------------------------------
    # Build base model
    # ------------------------------------------------------------------
    if args.pruner == 'ncp':
        U = _load_u_matrix(args.u_filepath)
        U_ab, U_ab_T = ablate_subspace_matrix(U, args.subspace_dims, args.irrelevant_subspaces)
        # AugmentedVGG16 loads ImageNet weights internally; if a custom checkpoint
        # is given those weights are overridden below. encode/decode are always
        # initialised from U (not from the checkpoint).
        model = AugmentedVGG16(U_ab, U_ab_T)
    else:
        # Skip downloading ImageNet weights when a custom checkpoint is provided.
        model = models.vgg16(pretrained=(args.pretrained_model_path is None))

    model.classifier[6] = nn.Linear(4096, 2)  # 2-class binary head

    # ------------------------------------------------------------------
    # Optionally load a pretrained checkpoint
    # ------------------------------------------------------------------
    if args.pretrained_model_path:
        print(f"Loading pretrained weights from: {args.pretrained_model_path}")
        sd = _load_checkpoint_state_dict(args.pretrained_model_path)
        if args.pruner == 'ncp':
            # Checkpoint uses vanilla VGG16 keys; remap to AugmentedVGG16 layout.
            # encode/decode keys are absent (set by U); strict=False tolerates this.
            sd = _remap_vgg16_features_to_augmented(sd)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            # Re-freeze encode/decode — load_state_dict(strict=False) won't have
            # touched them, but be explicit in case keys ever collide.
            for param in model.encode.parameters():
                param.requires_grad = False
            for param in model.decode.parameters():
                param.requires_grad = False
            if missing:
                print(f"  Missing keys (expected: encode/decode only): {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
        else:
            model.load_state_dict(sd, strict=True)
        print("  Pretrained weights loaded.")

    if args.cuda:
        print('cuda status: ', torch.cuda.is_available())
        model = model.cuda()

    from prune_vgg import PruningFineTuner, VanillaVGGAdapter, AugmentedVGGAdapter

    print("Initializing PruningFineTuner...")
    if args.pruner == 'ncp':
        adapter = AugmentedVGGAdapter(model)
    else:
        adapter = VanillaVGGAdapter(model)

    tuner = PruningFineTuner(args, model, adapter)

    # Persist the eval supplement filenames immediately so the exact evaluation
    # distribution is recorded even if the run crashes before saving the model.
    # Empty when min_male_with_attr=0 or the prune_val already had enough samples.
    os.makedirs(args.out_dir, exist_ok=True)
    supplement_log_path = os.path.join(args.out_dir, 'eval_supplement_fnames.txt')
    with open(supplement_log_path, 'w') as _f:
        _f.write(f"# Filenames pulled from official TEST split to supplement eval set\n")
        _f.write(f"# min_male_with_attr={args.min_male_with_attr}  "
                 f"count={len(tuner.eval_supplement_fnames)}\n")
        for fn in tuner.eval_supplement_fnames:
            _f.write(fn + '\n')
    print(f"Eval supplement log: {supplement_log_path} "
          f"({len(tuner.eval_supplement_fnames)} filenames)")

    print("Training new 2-d output layer...")
    # only train output layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    # Create optimizer after freezing so it is scoped to the trainable params only
    optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1): #debug: reduced from 15
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
    tuner.prune(fine_tune_conv_layers=args.fine_tune_conv_layers,
                fine_tune_without_augmented_layers=args.fine_tune_without_augmented_layers)

    ### Save model weights and mid-pruning metrics
    # Note: if augmented, encode/decode are removed prior to final fine-tuning
    # (on_before_final_finetune deletes them), so adapter.iter_modules_forward_order()
    # here reflects the post-deletion structure of the final model.
    pruned_structure = [
        m.out_channels
        for layer_idx, m in enumerate(adapter.iter_modules_forward_order())
        if isinstance(m, torch.nn.Conv2d) and adapter.is_prunable_conv(m, layer_idx)
    ]

    if args.pruner == 'ncp':
        out_path = os.path.join(args.out_dir, 'ncp-2.pth')
    else:
        out_path = os.path.join(args.out_dir, 'van.pth')
    os.makedirs(args.out_dir, exist_ok=True)
    # Save model + pruner metadata
    torch.save({
        'model': tuner.model,
        'state_dict': tuner.model.state_dict(),
        'pruned_structure': pruned_structure,
        'train_loss': tuner.train_loss_tot,
        'train_acc': tuner.train_acc_tot,
        'test_loss': tuner.test_loss_tot,
        'test_acc': tuner.test_acc_tot,
        'test_iter': tuner.test_iter,
        'test_precision_per_class': tuner.test_precision_tot,
        'test_recall_per_class': tuner.test_recall_tot,
        'subgroup_stats': tuner.subgroup_stats_tot,
    }, out_path)


if __name__ == "__main__":
    test_pruning_pipeline()
