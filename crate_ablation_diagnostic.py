"""
crate_ablation_diagnostic.py

Step 1 – warm up a vanilla VGG16 for N epochs on biased crate/packet data.
Step 2 – evaluate subgroup stats three ways at inference time:
   (a) no ablation  (vanilla forward)
   (b) ablate subspace 1
   (c) ablate subspace 3
   (d) ablate subspaces 1+3

This isolates which ablation best reduces the WM/NWM recall gap without
hurting overall accuracy, before committing to a full pruning run.
"""

import sys, argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import models, transforms
from sklearn.metrics import average_precision_score as _ap_score

sys.path.insert(0, '/n/fs/ncp/NCP.v2')
sys.path.insert(0, '/n/fs/ncp/watermark_imagenet')
from AugmentedVGG16 import AugmentedVGG16, ablate_subspace_matrix
from watermark_transform import AddWatermark
from run_watermark_pruning_experiment import (
    build_splits, make_loaders, EXPERIMENT_CONFIGS, SUBSPACE_DIMS,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def build_vanilla_model(cuda):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.classifier[6] = nn.Linear(4096, 2)
    if cuda:
        m = m.cuda()
    return m


def warmup(model, train_loader, epochs, lr, cuda):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier[6].parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.classifier[6].parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bidx, batch in enumerate(train_loader):
            x, y = batch[0], batch[1]
            if cuda:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if bidx % 20 == 0:
                print(f"  warmup e{ep+1} b{bidx} loss={loss.item():.4f}")
        print(f"  warmup epoch {ep+1}/{epochs}  avg_loss={total_loss/len(train_loader):.4f}")

    for p in model.parameters():
        p.requires_grad = True


def eval_subgroups(model, test_loader, cuda, label):
    """
    Collect per-image scores and group labels; compute AP per watermark group.

    For the ablation conditions (no fine-tuning after changing U), accuracy and
    threshold-based metrics (recall, precision) are unreliable because the
    classifier was calibrated on a different feature distribution.  AP is
    rank-based and therefore calibration-insensitive: it only depends on the
    relative ordering of output[:, 1] scores, which remains meaningful even
    after the augmented projection changes the absolute scale.

    Key diagnostic:
      AP_g0  — ranks NWM positives above NWM negatives  (legitimate features)
      AP_g1  — ranks WM  positives above WM  negatives  (watermark + features)
      AP_g1 - AP_g0 gap: positive = model uses watermark cue.
        Ablating the WM subspace should reduce this gap by dropping AP_g1
        toward AP_g0 without materially affecting AP_g0.
    """
    model.eval()
    ap_scores, ap_labels, ap_genders = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            data, target, gender = batch[0], batch[1], batch[2]
            if cuda:
                data = data.cuda(non_blocking=True)
            output = model(data)
            ap_scores.append(output[:, 1].cpu())
            ap_labels.append(target.view(-1))
            ap_genders.append(gender.view(-1))

    sc = torch.cat(ap_scores).numpy()
    lb = torch.cat(ap_labels).numpy()
    gn = torch.cat(ap_genders).numpy()

    ap_overall = float(_ap_score(lb, sc))
    ap_g0 = float(_ap_score(lb[gn == 0], sc[gn == 0])) if (gn == 0).sum() > 0 else float('nan')
    ap_g1 = float(_ap_score(lb[gn == 1], sc[gn == 1])) if (gn == 1).sum() > 0 else float('nan')
    ap_gap = ap_g1 - ap_g0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  AP_overall={ap_overall:.4f}")
    print(f"  AP_g0 (NWM)={ap_g0:.4f}  AP_g1 (WM)={ap_g1:.4f}  gap={ap_gap:+.4f}")
    print(f"  (acc/recall omitted — miscalibrated without fine-tuning after ablation)")

    model.train()
    return {
        'label': label,
        'ap_overall': ap_overall, 'ap_g0': ap_g0, 'ap_g1': ap_g1, 'ap_gap': ap_gap,
    }


def wrap_in_augmented(vanilla_model, U_ab, U_ab_T, cuda):
    """
    Build an AugmentedVGG16 that shares weights with vanilla_model.

    AugmentedVGG16 splits VGG's features into:
      self.before  = features[:23]   (up to ReLU after conv4_3)
      self.after   = features[23:]   (pool4 onward)
      self.classifier                (FC layers)
    Copy fine-tuned weights from vanilla_model into these submodules,
    then set augmented=True so the encode/decode path activates at inference.
    """
    aug = AugmentedVGG16(U_ab, U_ab_T)

    # Rebuild the state_dict keys: AugmentedVGG16.before.N.* ← features.N.*
    vgg_feat_sd = vanilla_model.features.state_dict()   # keys: "0.weight", "2.weight", ...
    before_sd, after_sd = {}, {}
    split = 23
    for key, val in vgg_feat_sd.items():
        idx = int(key.split('.')[0])
        if idx < split:
            before_sd[key] = val
        else:
            # renumber: after module 0 = features[23], so subtract split
            new_key = f"{idx - split}." + '.'.join(key.split('.')[1:])
            after_sd[new_key] = val

    aug.before.load_state_dict(before_sd)
    aug.after.load_state_dict(after_sd)
    # AugmentedVGG16 initialises classifier[6] as Linear(4096, 1000).
    # Replace with the correct output size before loading the fine-tuned weights.
    in_f = vanilla_model.classifier[6].in_features
    out_f = vanilla_model.classifier[6].out_features
    aug.classifier[6] = nn.Linear(in_f, out_f)
    aug.classifier.load_state_dict(vanilla_model.classifier.state_dict())
    aug.augmented = True
    if cuda:
        aug = aug.cuda()
    aug.eval()
    return aug


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',           type=int,   default=0)
    p.add_argument('--warmup_epochs',  type=int,   default=15)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--no_cuda',        action='store_true')
    p.add_argument('--out',            type=str,   default=None,
                   help='Path to save results dict (.pt)')
    args = p.parse_args()

    cuda = torch.cuda.is_available() and not args.no_cuda
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    cfg = EXPERIMENT_CONFIGS['crate']
    train_samples, rank_samples, test_samples = build_splits(cfg['pos_dir'], cfg['neg_dir'])
    add_wm = AddWatermark(image_size=224)
    train_loader, _, test_loader = make_loaders(
        train_samples, rank_samples, test_samples,
        add_wm, batch_size=args.batch_size, cuda=cuda,
    )

    # ── warmup vanilla model ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Warming up vanilla VGG16 for {args.warmup_epochs} epochs  (seed={args.seed})")
    print(f"{'='*60}")
    vanilla = build_vanilla_model(cuda)
    warmup(vanilla, train_loader, args.warmup_epochs, args.lr, cuda)

    # ── load U ──────────────────────────────────────────────────────────────
    U = torch.load(cfg['u_path'], map_location='cpu', weights_only=False).float()

    # ── evaluation conditions ───────────────────────────────────────────────
    results = []

    # (a) no ablation — vanilla forward path
    vanilla.eval()
    results.append(eval_subgroups(vanilla, test_loader, cuda, label='no ablation (vanilla)'))

    # (b–d) augmented forward path with each ablation
    for ablate_ss, label in [
        ([1],    'ablate subspace 1'),
        ([3],    'ablate subspace 3'),
        ([1, 3], 'ablate subspaces 1+3'),
    ]:
        U_ab, U_ab_T = ablate_subspace_matrix(U.clone(), SUBSPACE_DIMS, ablate_ss)
        if cuda:
            U_ab, U_ab_T = U_ab.cuda(), U_ab_T.cuda()
        aug = wrap_in_augmented(vanilla, U_ab, U_ab_T, cuda)
        results.append(eval_subgroups(aug, test_loader, cuda, label=label))

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (crate, seed={args.seed}, warmup={args.warmup_epochs} epochs)")
    print(f"  Primary metric: AP (rank-based, calibration-insensitive)")
    print(f"  AP_gap = AP_g1(WM) - AP_g0(NWM): ablating WM subspace should reduce this")
    print(f"{'='*60}")
    print(f"  {'condition':<25}  {'AP_overall':>10}  {'AP_g0(NWM)':>10}  {'AP_g1(WM)':>9}  {'AP_gap':>8}")
    for r in results:
        print(f"  {r['label']:<25}  {r['ap_overall']:>10.4f}  {r['ap_g0']:>10.4f}  "
              f"{r['ap_g1']:>9.4f}  {r['ap_gap']:>+8.4f}")

    if args.out:
        torch.save(results, args.out)
        print(f"\n[saved] {args.out}")


if __name__ == '__main__':
    main()
