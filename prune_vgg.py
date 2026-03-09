"""
prune_vgg.py

Unified pruning + fine-tuning pipeline for both vanilla VGG16 and AugmentedVGG16.
Replaces the separate prune_aug_vgg.py and prune_van_vgg.py files.

Architecture
------------
Model-specific behaviour is isolated behind VGGAdapterBase (Strategy pattern):

    VGGAdapterBase          – abstract interface
      VanillaVGGAdapter     – wraps torchvision VGG16 (model.features / model.classifier)
      AugmentedVGGAdapter   – wraps AugmentedVGG16 (model.before/encode/decode/after/classifier)

The shared FilterPruner and PruningFineTuner classes hold an adapter instance
and call adapter methods for every model-specific operation.

Key design notes
----------------
- Hook accumulation fix: FilterPruner.reset() stores RemovableHandles and removes
  them before re-registering, so fhook fires exactly once per module per forward pass.
- Model reference staleness fix: after prune_conv_layer_sequential creates a new model
  object, PruningFineTuner updates both self.model and self.adapter.model so all
  subsequent adapter calls operate on the latest model.
- Disallowed layers: computed programmatically from model structure in each adapter's
  _compute_disallowed_layer_indices(). These layers are excluded from filter_ranks during
  backward_lrp (only grad_index is incremented, no entry is written).
- Ranking with eval mode: get_candidates_to_prune() switches to model.eval() before the
  LRP ranking pass to disable VGG's dropout layers, then restores model.train().
- Per-class stats: test() accumulates per-class TP/FP/FN across all batches and appends
  precision/recall dicts to test_precision_tot, test_recall_tot per evaluation call.

Used by
-------
run_PFT.py — instantiates the adapter and PruningFineTuner, calls prune().
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models

from lrp import lrp
from prune_layer import prune_conv_layer_sequential
import data as dataset

from operator import itemgetter
from heapq import nsmallest
from sklearn.metrics import average_precision_score as _ap_score


# ---------------------------------------------------------------------------
# Forward hook (attached to every module so module.input / module.output exist)
# ---------------------------------------------------------------------------

def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class VGGAdapterBase:
    """
    Abstract adapter that isolates model-specific structure from FilterPruner
    and PruningFineTuner.  Concrete subclasses implement every abstract method.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        # Frozenset of layer indices (in the flattened conv_seq from
        # prune_layer.get_conv_seq_and_name) that must never be pruned.
        self._disallowed = self._compute_disallowed_layer_indices()

    # ------------------------------------------------------------------
    # Abstract – must be overridden
    # ------------------------------------------------------------------

    def _compute_disallowed_layer_indices(self) -> frozenset:
        """Return frozenset of layer indices that must not be pruned."""
        raise NotImplementedError

    def iter_modules_forward_order(self) -> list:
        """
        Flat ordered list of all modules (features + classifier for vanilla;
        before [+ encode/decode] + after + classifier for augmented).
        Used for forward-hook registration and building the backward LRP list.
        """
        raise NotImplementedError

    def forward_lrp(self, x: torch.Tensor):
        """
        Run a forward pass suitable for LRP.
        Returns (logits, activation_to_layer, num_activations) where:
          activation_to_layer: dict mapping activation_index (count of Conv2d
            layers seen) to the flattened sequential layer index used by
            prune_conv_layer_sequential.
          num_activations: total number of Conv2d layers counted.
        """
        raise NotImplementedError

    def _get_gamma_param(self, module_idx: int) -> float:
        """
        Per-module gamma value for the 'gamma heuristic' LRP schedule.
        module_idx counts from the output side (0 = first reversed module).
        """
        raise NotImplementedError

    def total_prunable_filters(self) -> int:
        """
        Count of all prunable conv output channels (excludes disallowed layers).
        Used by PruningFineTuner to compute the pruning iteration budget.
        """
        raise NotImplementedError

    def setup_iterative_finetune_params(self):
        """Set requires_grad for the iterative fine-tuning phase."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete defaults (may be overridden by subclasses)
    # ------------------------------------------------------------------

    def iter_modules_reverse_order(self) -> list:
        return list(reversed(self.iter_modules_forward_order()))

    def is_prunable_conv(self, module: nn.Module, layer_idx: int) -> bool:
        """
        True if the Conv2d at layer_idx is eligible for the pruning plan.
        Disallowed layers (conv4_3, encode, decode) return False.
        """
        return layer_idx not in self._disallowed

    def should_accumulate_rank(self, module: nn.Module) -> bool:
        """
        True if this Conv2d should be counted in grad_index (and potentially
        have its ranks accumulated) during backward_lrp.

        Returns False for modules that have NO entry in activation_to_layer
        (i.e. encode/decode in augmented models). This ensures grad_index
        counts exactly the same set of Conv2d layers that forward_lrp counted.

        Disallowed layers that ARE in activation_to_layer (e.g. conv4_3) still
        return True here — they are counted for grad_index purposes but are
        excluded from filter_ranks by the is_prunable_conv check inside
        backward_lrp.
        """
        return True

    def get_backward_lrp_rule(self, module: nn.Module, reverse_idx: int,
                               total_modules: int, method: str, param) -> tuple:
        """
        Return (lrp_var, lrp_param) to pass to lrp() for this module.

        reverse_idx: position in the reversed module list (0 = output side).
        The last position (reverse_idx == total_modules - 1) corresponds to the
        pixel / first conv layer, which uses the 'first' rule.
        """
        if reverse_idx == total_modules - 1:        # pixel layer → 'first' rule
            return ('first', None)
        if method == 'gamma' and param == 'heuristic':
            return ('gamma', self._get_gamma_param(reverse_idx))
        return (method, param)

    def on_before_final_finetune(self):
        """
        Hook called just before the final fine-tuning phase.
        Augmented adapter overrides this to disable augmented mode and delete
        encode/decode; vanilla is a no-op.
        """
        pass


# ---------------------------------------------------------------------------
# VanillaVGGAdapter
# ---------------------------------------------------------------------------

class VanillaVGGAdapter(VGGAdapterBase):
    """Adapter for a standard torchvision VGG16 (model.features + model.classifier)."""

    def _compute_disallowed_layer_indices(self) -> frozenset:
        """
        Disallow conv4_3 = the last Conv2d before the 4th MaxPool in model.features.
        conv4_3 is the designated concept-insertion split boundary (shared with
        AugmentedVGG16); its channel count must stay fixed to remain compatible
        with the U matrix projection dimensions across vanilla and augmented runs.
        """
        pool_count, last_conv = 0, None
        for layer, (_, m) in enumerate(self.model.features._modules.items()):
            if isinstance(m, nn.MaxPool2d):
                pool_count += 1
                if pool_count == 4:
                    return frozenset({last_conv}) if last_conv is not None else frozenset()
            if isinstance(m, nn.Conv2d):
                last_conv = layer
        return frozenset()

    def iter_modules_forward_order(self) -> list:
        return list(self.model.features) + list(self.model.classifier)

    def forward_lrp(self, x: torch.Tensor):
        in_size = x.size(0)
        a2l, idx = {}, 0
        for layer, (_, m) in enumerate(self.model.features._modules.items()):
            x = m(x)
            if isinstance(m, nn.Conv2d):
                a2l[idx] = layer
                idx += 1
        logits = self.model.classifier(x.view(in_size, -1))
        return logits, a2l, idx

    def _get_gamma_param(self, i: int) -> float:
        """
        Gamma schedule for vanilla VGG16 (counting from output side):
          classifier + Conv5   → 0.01
          Conv4                → 0.10
          Conv3                → 0.25
          Conv2 + Conv1        → 0.50
        """
        if i <= 6:    return 0.01   # classifier layers
        elif i <= 13: return 0.01   # Conv5
        elif i <= 20: return 0.10   # Conv4
        elif i <= 27: return 0.25   # Conv3
        else:         return 0.50   # Conv2, Conv1

    def total_prunable_filters(self) -> int:
        return sum(
            m.out_channels
            for layer, (_, m) in enumerate(self.model.features._modules.items())
            if isinstance(m, nn.Conv2d) and layer not in self._disallowed
        )

    def setup_iterative_finetune_params(self):
        for p in self.model.features.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# AugmentedVGGAdapter
# ---------------------------------------------------------------------------

class AugmentedVGGAdapter(VGGAdapterBase):
    """
    Adapter for AugmentedVGG16 (model.before / encode / decode / after / classifier).

    Disallowed layers: encode, decode, and conv4_3 (= last Conv2d in model.before).
    Encode and decode are also excluded from rank accumulation entirely via
    should_accumulate_rank(), since they have no entry in activation_to_layer.
    """

    def _compute_disallowed_layer_indices(self) -> frozenset:
        encode_idx = len(self.model.before)         # = 23 for standard VGG16 split
        decode_idx = len(self.model.before) + 1     # = 24
        # Last Conv2d in model.before = conv4_3 (index 21 for standard VGG16)
        last_conv = next(
            (i for i, m in reversed(list(enumerate(self.model.before)))
             if isinstance(m, nn.Conv2d)),
            None
        )
        indices = [encode_idx, decode_idx]
        if last_conv is not None:
            indices.append(last_conv)
        return frozenset(indices)

    def iter_modules_forward_order(self) -> list:
        mods = list(self.model.before)
        if self.model.augmented:
            mods += [self.model.encode, self.model.decode]
        mods += list(self.model.after)
        mods += list(self.model.classifier)
        return mods

    def forward_lrp(self, x: torch.Tensor):
        in_size = x.size(0)
        a2l, idx = {}, 0
        for layer, m in enumerate(self.model.before):
            x = m(x)
            if isinstance(m, nn.Conv2d):
                a2l[idx] = layer
                idx += 1
        offset = len(self.model.before)
        if self.model.augmented:
            x = self.model.encode(x)
            x = self.model.decode(x)
            offset += 2     # encode and decode are skipped in activation counting
        for layer, m in enumerate(self.model.after):
            x = m(x)
            if isinstance(m, nn.Conv2d):
                a2l[idx] = offset + layer
                idx += 1
        logits = self.model.classifier(x.view(in_size, -1))
        return logits, a2l, idx

    def should_accumulate_rank(self, module: nn.Module) -> bool:
        """
        Encode and decode are Conv2d layers but have no activation_to_layer entry.
        Return False for them so grad_index is not incremented and no bogus rank
        is written for an act_idx that doesn't exist in activation_to_layer.
        """
        if hasattr(self.model, 'encode') and (
                module is self.model.encode or module is self.model.decode):
            return False
        return True

    def get_backward_lrp_rule(self, module: nn.Module, reverse_idx: int,
                               total_modules: int, method: str, param) -> tuple:
        """
        Augmented-specific override:
          - pixel layer → 'first'
          - encode / decode → gamma(0.0) in ALL LRP modes (stable pass-through
            for frozen 1x1 concept-projection layers)
          - everything else → base class dispatch (gamma schedule or method/param)
        """
        if reverse_idx == total_modules - 1:
            return ('first', None)
        if hasattr(self.model, 'encode') and (
                module is self.model.encode or module is self.model.decode):
            # Stable pass-through: gamma with param=0 ≈ standard LRP through frozen 1x1 convs
            return ('gamma', 0.0)
        if method == 'gamma' and param == 'heuristic':
            return ('gamma', self._get_gamma_param(reverse_idx))
        return (method, param)

    def _get_gamma_param(self, i: int) -> float:
        """
        Gamma schedule for AugmentedVGG16 (counting from output side).
        Encode/decode (indices 15 and 16 in reversed order) are handled by
        get_backward_lrp_rule before this is called, so they don't appear here.
          classifier + Conv5       → 0.01
          aug 1x1 + Conv4          → 0.10  (enc/dec caught above)
          Conv3                    → 0.25
          Conv2 + Conv1            → 0.50
        """
        if i <= 6:              return 0.01   # classifier layers
        elif i <= 13:           return 0.01   # Conv5
        elif 14 <= i <= 22:     return 0.10   # aug 1x1 + Conv4
        elif 23 <= i <= 29:     return 0.25   # Conv3
        else:                   return 0.50   # Conv2, Conv1

    def total_prunable_filters(self) -> int:
        total = sum(
            m.out_channels for i, m in enumerate(self.model.before)
            if isinstance(m, nn.Conv2d) and i not in self._disallowed
        )
        offset = len(self.model.before) + (2 if self.model.augmented else 0)
        total += sum(
            m.out_channels for i, m in enumerate(self.model.after)
            if isinstance(m, nn.Conv2d) and (offset + i) not in self._disallowed
        )
        return total

    def setup_iterative_finetune_params(self):
        for p in self.model.before.parameters():
            p.requires_grad = True
        for p in self.model.encode.parameters():
            p.requires_grad = False     # encode is frozen (concept projection)
        for p in self.model.decode.parameters():
            p.requires_grad = False     # decode is frozen (concept projection)
        for p in self.model.after.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True

    def on_before_final_finetune(self):
        """
        Permanently disable augmented mode and remove encode/decode before
        the final fine-tuning pass so the model is pruned-VGG-only at the end.
        """
        self.model.augmented = False
        del self.model.encode
        del self.model.decode


# ---------------------------------------------------------------------------
# FilterPruner
# ---------------------------------------------------------------------------

class FilterPruner:
    """
    Registers forward hooks on model modules (via adapter), runs forward/backward
    LRP passes, accumulates per-filter relevance scores, and builds a pruning plan.
    """

    def __init__(self, adapter: VGGAdapterBase, args):
        self.adapter = adapter
        self.args = args
        self._hook_handles: list = []
        self.reset()

    def reset(self):
        """
        Clear accumulated filter ranks and re-register forward hooks.
        Removes all existing hooks before registering new ones to prevent
        accumulation across multiple calls (each call to get_candidates_to_prune
        calls reset() so without this hooks would fire N+1 times after N steps).
        """
        self.filter_ranks = {}
        self.grad_index = 0
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        for m in self.adapter.iter_modules_forward_order():
            self._hook_handles.append(m.register_forward_hook(fhook))

    def forward_lrp(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass via adapter. Stores activation_to_layer and activation_index."""
        self.grad_index = 0
        logits, self.activation_to_layer, self.activation_index = \
            self.adapter.forward_lrp(x)
        return logits

    def backward_lrp(self, R: torch.Tensor, relevance_method: str = 'alpha', param=1):
        """
        Propagate relevance R backwards through all modules.

        Supported modes:
          ('alpha', 1)           – positive relevance only
          ('gamma', 'heuristic') – per-layer gamma schedule from adapter

        For each Conv2d where should_accumulate_rank() is True:
          - grad_index is always incremented (keeps act_idx mapping consistent)
          - filter_ranks is only updated if is_prunable_conv() is True (disallowed
            layers like conv4_3 are counted for indexing but not ranked)

        LRP rule per module is determined by adapter.get_backward_lrp_rule().
        """
        if not ((relevance_method == 'gamma' and param == 'heuristic') or
                (relevance_method == 'alpha' and param == 1)):
            raise NotImplementedError(
                f"backward_lrp only supports ('alpha',1) and ('gamma','heuristic'), "
                f"got ({relevance_method!r}, {param!r})"
            )

        modules = self.adapter.iter_modules_reverse_order()
        total = len(modules)

        for i, module in enumerate(modules):
            if isinstance(module, nn.Conv2d) and self.adapter.should_accumulate_rank(module):
                # This conv IS in activation_to_layer; compute its activation index.
                act_idx = self.activation_index - self.grad_index - 1
                layer_idx = self.activation_to_layer.get(act_idx)

                # Only write to filter_ranks if the layer is prunable.
                # Disallowed layers (conv4_3) still increment grad_index but are
                # not added to filter_ranks, so they cannot appear in the pruning plan.
                if layer_idx is not None and self.adapter.is_prunable_conv(module, layer_idx):

                    values = R.abs().sum(dim=(0, 2, 3)).data #abs value to measure overall "importance" of a filter

                    if act_idx not in self.filter_ranks:
                        t = torch.FloatTensor(R.size(1)).zero_()
                        self.filter_ranks[act_idx] = t.cuda() if self.args.cuda else t
                    self.filter_ranks[act_idx] += values

                self.grad_index += 1    # always increment to keep act_idx correct

            lrp_var, lrp_param = self.adapter.get_backward_lrp_rule(
                module, i, total, relevance_method, param)
            R = lrp(module, R.data, lrp_var=lrp_var, param=lrp_param)

    def normalize_ranks_per_layer(self):
        """
        Normalize accumulated relevance scores within each layer so they sum to 1,
        enabling cross-layer rank comparisons during filter selection.
        """
        for i in self.filter_ranks:
            if self.args.relevance:
                v = self.filter_ranks[i]
                self.filter_ranks[i] = (v / v.sum()).cpu()

    def lowest_ranking_filters(self, num: int) -> list:
        """
        Return the `num` (layer_idx, filter_idx, score) tuples with the smallest
        accumulated relevance.  filter_ranks only contains prunable-layer entries
        (disallowed layers were excluded at accumulation time), so the
        is_prunable_conv check here is a defensive double-guard.
        """
        data = []
        modules = self.adapter.iter_modules_forward_order()
        for act_idx in sorted(self.filter_ranks.keys()):
            layer_idx = self.activation_to_layer[act_idx]
            module = modules[layer_idx] if layer_idx < len(modules) else None
            if not self.adapter.is_prunable_conv(module, layer_idx):
                continue    # defensive: should not trigger if backward_lrp is correct
            for j in range(self.filter_ranks[act_idx].size(0)):
                data.append((layer_idx, j, self.filter_ranks[act_idx][j]))
        return nsmallest(num, data, itemgetter(2))

    def get_pruning_plan(self, num_filters_to_prune: int) -> list:
        """
        Build a pruning plan of (layer_idx, filter_idx) pairs, adjusting filter
        indices to account for filters removed earlier in the same layer.
        Returns list of (layer_idx, adjusted_filter_idx).
        """
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # Group by layer
        filters_per_layer = {}
        for (l, f, _) in filters_to_prune:
            filters_per_layer.setdefault(l, []).append(f)

        # Sort and adjust indices: pruning filter i shifts all subsequent indices down by 1
        for l in filters_per_layer:
            filters_per_layer[l] = sorted(filters_per_layer[l])
            for i in range(len(filters_per_layer[l])):
                filters_per_layer[l][i] -= i

        result = []
        for l in filters_per_layer:
            for f in filters_per_layer[l]:
                result.append((l, f))
        return result


# ---------------------------------------------------------------------------
# PruningFineTuner
# ---------------------------------------------------------------------------

class PruningFineTuner:
    """
    Orchestrates the full prune-and-fine-tune loop.

    Accepts an adapter that encapsulates model-specific behaviour so this class
    works identically for vanilla and augmented VGG16.
    """

    def __init__(self, args, model: nn.Module, adapter: VGGAdapterBase):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.args = args
        self.model = model
        self.adapter = adapter

        self.setup_dataloaders()

        self.criterion = nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.adapter, args)
        self.model.train()
        self.save_loss = True

        # Tracking lists reset by prune(); initialized here so test() and
        # train_epoch() can be called before prune() without AttributeError.
        self.train_loss_tot: list = []
        self.test_loss_tot: list = []
        self.test_acc_tot: list = []
        self.test_iter: list = []
        self.num_param: list = []
        self.data_tot: list = []
        self.time_tot: list = []
        self.test_precision_tot: list = []   # list[dict[class_idx -> float]]
        self.test_recall_tot: list = []
        self.train_acc_tot: list = []        # training accuracy at end of each fine-tune block

        # Per-(gender × target) subgroup stats (populated by test(compute_subgroup=True))
        # Each entry is a dict with keys: acc_g{g}y{y}, n_g{g}y{y},
        # precision_g{g}, recall_g{g}   (g=0 Female / g=1 Male; y=0/1 attr)
        self.subgroup_stats_tot: list = []

    def setup_dataloaders(self):
        # Use spawn context to avoid CUDA+fork deadlock.
        # Default fork context inherits the parent's CUDA runtime when workers are created;
        # spawned workers start fresh Python interpreters with no CUDA state, so they can
        # safely initialize their own CUDA contexts. This allows num_workers>0 (prefetching)
        # without deadlocking even after model.cuda() has been called in the parent.
        if self.args.cuda:
            kwargs = {
                'num_workers': 3,
                'pin_memory': True,
                'multiprocessing_context': 'spawn',
            }
        else:
            kwargs = {}

        data_type = self.args.data_type.lower()
        if data_type == 'celeba_lipstick':
            train_dataset, eval_dataset, supplement_fnames = \
                dataset.get_celeba_attribute_splits_for_pruning(
                    min_male_with_attr=getattr(self.args, 'min_male_with_attr', 0),
                )
        else:
            raise ValueError(f"Unknown data_type: {self.args.data_type!r}")

        # Filenames pulled from downstream pool to supplement the eval set.
        # Empty list when min_male_with_attr=0 or target was already met.
        # Persist via run_PFT.py so the exact eval distribution is reproducible.
        self.eval_supplement_fnames: list = supplement_fnames

        print(f"train_dataset:{len(train_dataset)}, eval_dataset:{len(eval_dataset)}")

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.args.train_batch_size,
            shuffle=True, **kwargs)
        # test_loader / eval_loader: same DataLoader; yields (x, y, g) for
        # celeba data_types. test() ignores g; test_subgroup() uses it.
        self.test_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset, batch_size=self.args.test_batch_size,
            shuffle=False, **kwargs)
        self.eval_loader = self.test_loader
        self.train_num = len(self.train_loader)
        self.test_num = len(self.test_loader)

    def test(self, compute_subgroup: bool = False):
        """
        Evaluate on test_loader in a single forward pass.

        compute_subgroup: if True, also accumulate per-(gender × target) subgroup
          statistics in the same pass (no extra data reads).  Only meaningful for
          celeba-based data_types where batches yield (x, y, g).  Results are
          appended to self.subgroup_stats_tot.  Use this flag at post-recovery
          checkpoints in prune() so that overall and subgroup stats share one pass.
        """
        self.model.eval()
        print("[DEBUG test()] entered test()", flush=True)
        test_loss = 0
        correct = 0
        tp, fp, fn = {}, {}, {}

        # Subgroup accumulators — only populated when compute_subgroup=True
        _do_sg = compute_subgroup and self.args.data_type.lower() in ('celeba_lipstick',)
        sg_correct, sg_total = {}, {}
        sg_tp_g, sg_fp_g, sg_fn_g = {}, {}, {}
        _ap_scores, _ap_labels, _ap_genders = [], [], []  # for per-gender AP

        print(f"[DEBUG test()] creating DataLoader iterator (num_workers={self.test_loader.num_workers}, mp_context={self.test_loader.multiprocessing_context})", flush=True)
        _test_iter = iter(self.test_loader)
        print("[DEBUG test()] DataLoader iterator created, requesting first batch", flush=True)
        _first_batch = next(_test_iter, None)
        print(f"[DEBUG test()] first batch received: {None if _first_batch is None else [t.shape for t in _first_batch]}", flush=True)

        import itertools as _itertools
        for batch in _itertools.chain([_first_batch] if _first_batch is not None else [], _test_iter):
            data, target = batch[0], batch[1]
            gender = batch[2] if _do_sg and len(batch) > 2 else None
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = self.model(data)

            test_loss += self.criterion(output, target).item() * data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            # Per-class TP/FP/FN accumulation
            pred_flat = pred.view(-1).cpu()
            target_flat = target.data.view(-1).cpu()
            num_classes = output.size(1)
            for c in range(num_classes):
                pred_c = (pred_flat == c)
                target_c = (target_flat == c)
                tp[c] = tp.get(c, 0) + int((pred_c & target_c).sum())
                fp[c] = fp.get(c, 0) + int((pred_c & ~target_c).sum())
                fn[c] = fn.get(c, 0) + int((~pred_c & target_c).sum())

            # Per-subgroup accumulation (same pass, no extra I/O)
            if gender is not None:
                gender_cpu = gender.view(-1).cpu()
                _ap_scores.append(output[:, 1].detach().cpu())
                _ap_labels.append(target_flat)
                _ap_genders.append(gender_cpu)
                for p, y, g in zip(pred_flat.tolist(), target_flat.tolist(),
                                   gender_cpu.tolist()):
                    y, g = int(y), int(g)
                    key = (g, y)
                    sg_total[key]   = sg_total.get(key, 0) + 1
                    sg_correct[key] = sg_correct.get(key, 0) + (1 if p == y else 0)
                    if y == 1:
                        if p == 1: sg_tp_g[g] = sg_tp_g.get(g, 0) + 1
                        else:      sg_fn_g[g] = sg_fn_g.get(g, 0) + 1
                    else:
                        if p == 1: sg_fp_g[g] = sg_fp_g.get(g, 0) + 1

        test_loss /= len(self.test_loader.dataset)
        num_classes = len(tp)

        precision_d, recall_d = {}, {}
        for c in range(num_classes):
            precision_d[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
            recall_d[c]    = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        for c in range(num_classes):
            print(f'  Class {c}: Precision={precision_d[c]:.4f}  Recall={recall_d[c]:.4f}')

        if self.save_loss:
            self.test_acc_tot.append(
                100. * correct / len(self.test_loader.dataset))
            self.test_loss_tot.append(test_loss)
            self.test_iter.append(self.niter)
            self.test_precision_tot.append(precision_d)
            self.test_recall_tot.append(recall_d)
            self.train_acc_tot.append(getattr(self, '_last_train_acc', float('nan')))

        if _do_sg and sg_total:
            _nan = float('nan')
            stats = {'niter': self.niter}
            all_genders = sorted({k[0] for k in sg_total})
            for g in all_genders:
                tp_g = sg_tp_g.get(g, 0); fp_g = sg_fp_g.get(g, 0); fn_g = sg_fn_g.get(g, 0)
                stats[f'precision_g{g}'] = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else _nan
                stats[f'recall_g{g}']    = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else _nan
            for (g, y), tot in sg_total.items():
                stats[f'acc_g{g}y{y}'] = sg_correct.get((g, y), 0) / tot if tot > 0 else _nan
                stats[f'n_g{g}y{y}']   = tot
            if _ap_scores:
                _sc = torch.cat(_ap_scores).numpy()
                _lb = torch.cat(_ap_labels).numpy()
                _gn = torch.cat(_ap_genders).numpy()
                stats['ap_overall'] = float(_ap_score(_lb, _sc))
                for _g in all_genders:
                    _mask = (_gn == _g)
                    if _mask.sum() > 0 and _lb[_mask].sum() > 0:
                        stats[f'ap_g{_g}'] = float(_ap_score(_lb[_mask], _sc[_mask]))
                    else:
                        stats[f'ap_g{_g}'] = _nan
            LABEL = {(0, 0): "F-no-makeup", (0, 1): "F-makeup",
                     (1, 0): "M-no-makeup", (1, 1): "M-makeup"}
            print(f"  Subgroup eval (iter {self.niter}):")
            for (g, y), tot in sorted(sg_total.items()):
                acc = stats.get(f'acc_g{g}y{y}', _nan)
                print(f"    {LABEL.get((g, y), f'g{g}y{y}')}: n={tot}  "
                      f"acc={'nan' if acc != acc else f'{acc:.4f}'}")
            for g in all_genders:
                p = stats.get(f'precision_g{g}', _nan); r = stats.get(f'recall_g{g}', _nan)
                ap = stats.get(f'ap_g{g}', _nan)
                print(f"    {'Male' if g else 'Female'}: "
                      f"prec={'nan' if p != p else f'{p:.4f}'}  "
                      f"rec={'nan' if r != r else f'{r:.4f}'}  "
                      f"AP={'nan' if ap != ap else f'{ap:.4f}'}")
            if 'ap_overall' in stats:
                print(f"    Overall AP: {stats['ap_overall']:.4f}")
            self.subgroup_stats_tot.append(stats)

        self.model.train()

    def train_epoch(self, optimizer=None, rank_filters: bool = False):
        self.train_loss_batch = 0
        self._train_correct = 0
        self._train_total = 0

        # Optional cap on samples used for ranking (keeps LRP ranking cost bounded)
        rank_cap = getattr(self.args, 'rank_n', None) if rank_filters else None
        seen = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.train_batch(optimizer, batch_idx, data, target, rank_filters)

            if rank_cap is not None:
                seen += data.size(0)
                if seen >= rank_cap:
                    break

        if self.save_loss:
            denom = (seen if (rank_cap is not None and seen > 0)
                     else len(self.train_loader.dataset))
            self.train_loss_tot.append(self.train_loss_batch / denom)

        if not rank_filters:
            self._last_train_acc = (self._train_correct / self._train_total
                                    if self._train_total > 0 else float('nan'))

    def train_batch(self, optimizer, batch_idx, batch, label, rank_filters: bool):
        self.model.zero_grad()

        if rank_filters:
            if self.args.relevance:   # LRP-based ranking
                output = self.pruner.forward_lrp(batch)

                T = torch.zeros_like(output)
                for ii in range(len(label)):
                    T[ii, label[ii]] = 1.0

                self.pruner.backward_lrp(T.data)

                print('Train Epoch: [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(batch), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader)))
                loss = self.criterion(output, label)
                self.train_loss_batch += loss.item()
            else:
                raise NotImplementedError("Gradient-based filter ranking not implemented")
        else:
            output = self.model(batch)
            loss = self.criterion(output, label)
            loss.backward()
            optimizer.step()
            # Accumulate training accuracy using the already-computed output (no extra forward pass)
            pred = output.detach().max(1, keepdim=True)[1]
            self._train_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
            self._train_total += label.size(0)
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(batch), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
            self.train_loss_batch += loss.item()

    def total_num_filters(self) -> int:
        return self.adapter.total_prunable_filters()

    def train(self, optimizer=None, epoches: int = 10):
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.classifier.parameters(),
                lr=self.args.lr, momentum=self.args.momentum)
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
        print("Finished fine tuning")

    def get_candidates_to_prune(self, num_filters_to_prune: int) -> list:
        """
        Run a ranking pass (LRP) and return the pruning plan.
        Uses model.eval() during ranking to disable VGG's dropout layers,
        ensuring filter ranks are deterministic.
        """
        self.pruner.reset()
        self.model.eval()                           # disable dropout for stable LRP ranks
        self.train_epoch(rank_filters=True)
        self.model.train()
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filters_to_prune)

    def prune(self, fine_tune_conv_layers: bool = True,
              fine_tune_without_augmented_layers: bool = False,
              final_finetune_epochs: int = 5):
        """
        Full iterative prune-and-fine-tune loop.

        fine_tune_conv_layers: if True, update conv layers during fine-tuning
            (otherwise only classifier is updated).
        fine_tune_without_augmented_layers: debug flag for augmented models;
            temporarily disables the augmented path during each fine-tune step.
        """
        self.train_loss_tot = []
        self.test_loss_tot = []
        self.test_acc_tot = []
        self.test_iter = []
        self.num_param = []
        self.data_tot = []
        self.time_tot = []
        self.test_precision_tot = []
        self.test_recall_tot = []
        self.train_acc_tot = []
        self.subgroup_stats_tot = []
        self.save_loss = True

        self.niter = 0
        self.temp = 0
        self.model.train()

        print("Starting iterative prune-fine-tune loop.")
        # Set requires_grad via adapter for the iterative pruning phase
        self.adapter.setup_iterative_finetune_params()

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(number_of_filters * self.args.pr_step)
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        iterations = int(iterations * self.args.total_pr)

        print(f"Number of pruning iterations to reduce {self.args.total_pr*100:.0f}% "
              f"filters: {iterations}")

        # Baseline eval before any pruning (niter=0)
        # Disable augmented path so stats reflect the prunable conv layers only,
        # consistent with the final eval (where encode/decode are already deleted).
        print("Baseline eval (before pruning):", flush=True)
        _was_augmented = getattr(self.model, 'augmented', False)
        if _was_augmented:
            self.model.augmented = False
        self.test(compute_subgroup=True)
        if _was_augmented:
            self.model.augmented = True

        for kk in range(iterations):
            print("Ranking filters.. {}".format(kk))
            self.niter += 1
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)

            layers_pruned = {}
            for layer_index, filter_index in prune_targets:
                layers_pruned[layer_index] = layers_pruned.get(layer_index, 0) + 1
            print("Layers that will be pruned", layers_pruned)
            print("Pruning filters..")

            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_conv_layer_sequential(
                    model, layer_index, filter_index, cuda_flag=self.args.cuda)

            self.model = model.cuda() if self.args.cuda else model
            # Keep adapter's model reference in sync to avoid staleness
            self.adapter.model = self.model

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters remaining", str(message))

            print("Fine tuning to recover from pruning iteration.")
            # Debug: temporarily disable augmented path during fine-tuning if requested
            if fine_tune_without_augmented_layers and hasattr(self.model, 'augmented'):
                was_augmented = self.model.augmented
                self.model.augmented = False
            if fine_tune_conv_layers:
                optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.args.lr, momentum=self.args.momentum)
            else:
                optimizer = None
            try:
                self.train(optimizer, epoches=2)
            finally:
                if fine_tune_without_augmented_layers and hasattr(self.model, 'augmented'):
                    self.model.augmented = was_augmented
            # Single eval pass at end of each (prune + fine-tune) iteration.
            # Disable augmented path so stats reflect pruned conv layers only.
            _was_augmented = getattr(self.model, 'augmented', False)
            if _was_augmented:
                self.model.augmented = False
            self.test(compute_subgroup=True)
            if _was_augmented:
                self.model.augmented = True

        print("Finished. Removing augmented layers (if any) and doing final fine-tune.")
        self.niter += 1

        # Pre-finetune hook: aug disables augmented mode and removes encode/decode
        self.adapter.on_before_final_finetune()

        # Enable all remaining parameters for final fine-tuning
        # (encode/decode are already deleted for aug, so model.parameters() is clean)
        for param in self.model.parameters():
            param.requires_grad = True

        # Create a fresh optimizer: model structure may have changed (encode/decode deleted)
        # and any previous optimizer holds stale parameter references.
        if final_finetune_epochs > 0:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr, momentum=self.args.momentum)
            self.train(optimizer, epoches=final_finetune_epochs)
        self.test(compute_subgroup=True)
