# Concept-Based Pruning Pipeline (Augmented VGG16 + LRP)

This repo implements an iterative **pruning + fine-tuning (PFT)** pipeline for VGG16, with an optional
**concept-based “virtual layer” augmentation** inserted at `conv4_3`. Filters are ranked for pruning using
**Layer-wise Relevance Propagation (LRP)** and removed structurally (channel surgery).

---

## 1) Core idea: virtual concept layer at conv4_3

At `conv4_3`, VGG16 produces activations:

- `a ∈ R^{B×512×H×W}`

We insert two **frozen 1×1 Conv2d** layers:

- `encode` with weights `U^T`
- `decode` with weights `U`

Since a 1×1 conv is a linear map across channels at each spatial location, this implements:

- `z = U^T a`  (encode into concept coordinates)
- `a_hat = U z = U U^T a`  (decode back)

If we “ablate” concept subspaces by zeroing blocks of `U` and `U^T`, then downstream layers see
approximately a projection where those concept directions are removed.

Implementation:
- `AugmentedVGG16.py` constructs `before = features[:23]`, `after = features[23:]` and inserts
  `encode/decode` between them when `model.augmented=True`.

---

## 2) What gets pruned?

We prune **convolutional filters** (output channels) across the VGG feature extractor.

Two mechanisms exist:
- **Mask pruning** (keeps shapes): `prune_conv_layer()` via `torch.nn.utils.prune.custom_from_mask`
- **Structural pruning** (changes shapes): `prune_conv_layer_sequential()`
  - rebuilds the current Conv2d with `out_channels-1`
  - rebuilds the next Conv2d with `in_channels-1`
  - if pruning the last conv, shrinks the first Linear layer accordingly

In this project we use **structural pruning** during the main PFT loop.

Implementation:
- `prune_layer.py`

---

## 3) Ranking criterion: LRP relevance → per-filter scores

Filters are ranked by relevance computed via LRP, accumulated per-filter:

For a conv relevance tensor `R ∈ R^{B×C×H×W}`, we compute a per-filter scalar (one per output channel),
typically by summing over batch + spatial dimensions.

Ranking pass behavior:
- Run a forward pass with hooks storing `module.input` and `module.output`
- Backpropagate relevance from the output to every conv layer using `lrp.py`
- Accumulate relevance per filter across a subset of training samples
- Normalize within layer and prune globally smallest filters

Implementation:
- `lrp.py` (LRP rules)
- `prune_aug_vgg.py` / `prune_van_vgg.py` (FilterPruner.forward_lrp/backward_lrp)

---

## 4) Iterative PFT loop (the main experiment)

Each pruning run follows:

1. Evaluate baseline on `prune_val` (called `test_loader` in code)
2. Repeat for `K` iterations until reaching `total_pr` pruning fraction:
   a) Rank filters via an LRP “ranking epoch” on `ft_train`
   b) Select lowest-ranked filters (global, skipping DISALLOWED_LAYERS)
   c) Structurally prune selected filters using `prune_conv_layer_sequential`
   d) Fine-tune to recover performance for a few epochs
   e) Evaluate on `prune_val`

3. (Augmented only) remove augmentation for the final stage:
   - set `model.augmented=False`
   - delete `model.encode` and `model.decode`
   - fine-tune again on the pruned architecture

Implementation:
- `prune_aug_vgg.py` / `prune_van_vgg.py` (PruningFineTuner.prune)
- `run_PFT*.py` scripts orchestrate configuration, model init, and saving

---

## 5) CelebA data policy (important for fairness experiments)

CelebA official split counts:
- train: 162,770
- val:   19,867
- test:  19,962

For our fairness-through-pruning experiments:
- We do NOT rebalance protected groups (e.g., gender). We want pruning to reflect the naturally
  occurring correlations, otherwise fine-tuning confounds the debiasment effect.
- We avoid class balancing that would dramatically distort group prevalence (e.g., rare `(y=1,g=1)`
  strata), unless explicitly designed for a specific ablation study.

We instead:
- Use enough samples so empirical prevalence approximates the true training distribution.
- Split the official TRAIN split into three reproducible internal sets:
  1) `downstream_test` (withheld, never used during pruning; saved for later evaluation)
  2) `ft_train` (used for ranking + fine-tuning during pruning)
  3) `prune_val` (used as “test” during pruning iterations)

All splits must be reproducible by seed and fraction (and ideally by filename-based hashing),
so downstream evaluation can recreate the exact withheld subset.

Implementation:
- `celeba.py` defines a minimal indexing + dataset wrapper returning `(x,y,g,fname)`
- `data.py` defines split constructors and wraps datasets with `XYOnly` to expose `(x,y)` only

---

## 6) Reproducibility checklist

- Fix `args.seed` and use it for:
  - `torch.manual_seed`
  - `torch.Generator().manual_seed(seed)` for `random_split`
  - deterministic split functions (preferred for downstream recreation)
- Save metadata:
  - split fractions/seed (or list of withheld filenames)
  - pruning hyperparameters: `pr_step`, `total_pr`, ranking cap (`rank_n`), LRP rule
  - model type (augmented vs vanilla), U path + subspace ablations

---

## 7) File map

- `AugmentedVGG16.py` — inserts concept layer via frozen 1×1 convs (U^T then U)
- `lrp.py`            — LRP propagation rules (Linear/Conv/Pooling + special first-layer rule)
- `prune_layer.py`    — structural pruning surgery for VGG-style sequential convnets
- `prune_aug_vgg.py`  — ranking + pruning + fine-tune loop for AugmentedVGG16
- `prune_van_vgg.py`  — same but for vanilla torchvision VGG16
- `data.py`           — dataset constructors + transformations + split logic
- `celeba.py`         — CelebA index + dataset that returns `(x,y,g,fname)`
- `run_PFT*.py`       — experiment entrypoints (paths, args, model init, saving)

---

### Code Acknowledgements
Includes code adapted and taken from https://github.com/facebookresearch/Whac-A-Mole , https://github.com/seulkiyeom/LRP_pruning , https://github.com/p16i/disentangling-explanations .
