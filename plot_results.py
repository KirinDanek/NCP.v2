#!/usr/bin/env python3
"""
Ablation study plots — mean ± 1σ over seeds, one figure per metric group.

Produces two sets of 4 plots (ftwoaug and ftwaug conditions), with the
vanilla baseline shown in both:
  pruning_plots_ftwoaug_accuracy.png   pruning_plots_ftwaug_accuracy.png
  pruning_plots_ftwoaug_recall.png     pruning_plots_ftwaug_recall.png
  pruning_plots_ftwoaug_precision.png  pruning_plots_ftwaug_precision.png
  pruning_plots_ftwoaug_ap.png         pruning_plots_ftwaug_ap.png

Shaded bands show ±1 standard deviation across seeds.

Usage:
    python plot_results.py [--ablation_dir DIR] [--seeds 0 1 2 3 4] [--out base.png]
"""

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch

# One colour per model group; gender distinguished by line style
GROUP_COLORS = {
    'sx4':     'steelblue',
    'sx8':     'darkorange',
    'sx16':    'seagreen',
    'vanilla': 'crimson',
}
GENDER_STYLE = {0: '-',  1: '--'}   # Female solid, Male dashed
GENDER_LABEL = {0: 'Female', 1: 'Male'}
MS         = 4
ALPHA_BAND = 0.18

CONDITION_TITLE = {
    'ftwoaug': 'Fine-Tuned Without Augmented Layers',
    'ftwaug':  'Fine-Tuned With Augmented Layers',
}


# ── data helpers ──────────────────────────────────────────────────────────────

def _load(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu', weights_only=False)


def _sg_metric(d, key):
    """subgroup_stats list → {niter: float}"""
    return {s['niter']: s.get(key, float('nan'))
            for s in d.get('subgroup_stats', [])}


def _acc_metric(d):
    """test_acc (already %) aligned with test_iter → {niter: float}"""
    return {int(n): float(v)
            for n, v in zip(d.get('test_iter', []), d.get('test_acc', []))}


def _agg(dicts):
    """
    [{niter: float}, ...] → (sorted_niters, means, stds)
    NaN values are excluded from each niter's aggregate.
    """
    all_niters = sorted({n for d in dicts for n in d})
    means, stds = [], []
    for n in all_niters:
        vals = [d[n] for d in dicts
                if n in d and not math.isnan(float(d[n]))]
        means.append(float(np.mean(vals)) if vals else float('nan'))
        stds.append(float(np.std(vals))   if vals else float('nan'))
    return all_niters, means, stds


def _scale100(d):
    """Multiply all values in a {niter: float} dict by 100 (fraction → %)."""
    return {n: v * 100 for n, v in d.items()}


# ── plot helpers ──────────────────────────────────────────────────────────────

def _plot_ms(ax, niters, means, stds, color, ls, label):
    """Mean line + ±1σ shaded band, skipping NaN points."""
    pts = [(n, m, s) for n, m, s in zip(niters, means, stds)
           if not math.isnan(m)]
    if not pts:
        return
    ns, ms, ss = zip(*pts)
    ns, ms, ss = list(ns), list(ms), list(ss)
    ax.plot(ns, ms, color=color, ls=ls, marker='o', ms=MS, label=label)
    ax.fill_between(ns, [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)],
                    color=color, alpha=ALPHA_BAND)


def _gender_legend(ax):
    f_line = mlines.Line2D([], [], color='grey', ls='-',  label='── Female')
    m_line = mlines.Line2D([], [], color='grey', ls='--', label='-- Male')
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [f_line, m_line], l + ['── Female', '-- Male'],
              fontsize=7, loc='best')


def _finish(fig, suptitle, out_path):
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


def _out(base, *suffixes):
    root, ext = os.path.splitext(base)
    return f'{root}_{"_".join(suffixes)}{ext}'


# ── per-figure plot functions ─────────────────────────────────────────────────

def plot_accuracy(groups, suptitle, out_path):
    """groups: list of (label, color, [paths])"""
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, paths in groups:
        ds = [d for p in paths if (d := _load(p)) is not None]
        if not ds:
            continue
        niters, means, stds = _agg([_acc_metric(d) for d in ds])
        _plot_ms(ax, niters, means, stds, color=color, ls='-', label=label)
    ax.set_title('Test Accuracy vs Pruning Iteration', fontsize=11)
    ax.set_xlabel('Pruning iteration')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    _finish(fig, suptitle, out_path)


def _plot_2panel(groups, suptitle, out_path, key_base,
                 title_main, title_gap, ylabel_main, ylabel_gap):
    fig, (ax_main, ax_gap) = plt.subplots(1, 2, figsize=(12, 5))

    for label, color, paths in groups:
        ds = [d for p in paths if (d := _load(p)) is not None]
        if not ds:
            continue

        # Per-gender lines
        for g in (0, 1):
            dicts_g = [_scale100(_sg_metric(d, f'{key_base}_g{g}')) for d in ds]
            niters, means, stds = _agg(dicts_g)
            _plot_ms(ax_main, niters, means, stds,
                     color=color, ls=GENDER_STYLE[g],
                     label=f'{label} – {GENDER_LABEL[g]}')

        # Gap: compute per seed, then aggregate
        gap_dicts = []
        for d in ds:
            gap_d = {}
            for s in d.get('subgroup_stats', []):
                v0 = s.get(f'{key_base}_g0', float('nan'))
                v1 = s.get(f'{key_base}_g1', float('nan'))
                if not (math.isnan(v0) or math.isnan(v1)):
                    gap_d[s['niter']] = abs(v0 - v1) * 100
            gap_dicts.append(gap_d)
        niters, means, stds = _agg(gap_dicts)
        _plot_ms(ax_gap, niters, means, stds,
                 color=color, ls='-', label=label)

    ax_main.set_title(title_main, fontsize=11)
    ax_main.set_xlabel('Pruning iteration')
    ax_main.set_ylabel(ylabel_main)
    ax_main.set_ylim(0, 105)

    ax_gap.set_title(title_gap, fontsize=11)
    ax_gap.set_xlabel('Pruning iteration')
    ax_gap.set_ylabel(ylabel_gap)
    ax_gap.set_ylim(bottom=0)

    for ax in (ax_main, ax_gap):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    _gender_legend(ax_main)
    _finish(fig, suptitle, out_path)


# ── auto-discovery & top-level driver ────────────────────────────────────────

def _build_groups(ablation_dir, condition, seeds):
    """
    Returns list of (label, color, [paths]) for the given condition.
    Vanilla is always appended as the shared baseline.
    """
    groups = []
    for sx in ('4', '8', '16'):
        paths = [os.path.join(ablation_dir,
                              f'ncp-sx{sx}-seed{s}-{condition}',
                              'ncp-2.pth')
                 for s in seeds]
        if any(os.path.exists(p) for p in paths):
            groups.append((f'NCP sx{sx}', GROUP_COLORS[f'sx{sx}'], paths))

    vanilla_paths = [os.path.join(ablation_dir, f'vanilla-seed{s}', 'van.pth')
                     for s in seeds]
    if any(os.path.exists(p) for p in vanilla_paths):
        groups.append(('Vanilla', GROUP_COLORS['vanilla'], vanilla_paths))

    return groups


def plot_condition(ablation_dir, condition, seeds, out_base):
    groups   = _build_groups(ablation_dir, condition, seeds)
    suptitle = (f'CelebA Wearing_Lipstick – {CONDITION_TITLE[condition]} '
                f'(n={len(seeds)} seeds)')
    base     = _out(out_base, condition)

    plot_accuracy(groups, suptitle, _out(base, 'accuracy'))
    _plot_2panel(groups, suptitle, _out(base, 'recall'),
                 'recall',
                 'Per-Gender Recall vs Pruning Iteration',
                 'Recall Gap |Female − Male| vs Pruning Iteration',
                 'Recall (%)', '|Recall gap| (pp)')
    _plot_2panel(groups, suptitle, _out(base, 'precision'),
                 'precision',
                 'Per-Gender Precision vs Pruning Iteration',
                 'Precision Gap |Female − Male| vs Pruning Iteration',
                 'Precision (%)', '|Precision gap| (pp)')
    _plot_2panel(groups, suptitle, _out(base, 'ap'),
                 'ap',
                 'Per-Gender Average Precision vs Pruning Iteration',
                 'AP Gap |Female − Male| vs Pruning Iteration',
                 'Average Precision (%)', '|AP gap| (pp)')


# ── entry point ───────────────────────────────────────────────────────────────

_RESULTS      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
_ABLATION_DIR = os.path.join(_RESULTS, 'ablation')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ablation_dir', default=_ABLATION_DIR,
                        help=f'Directory containing ablation result subdirs '
                             f'(default: {_ABLATION_DIR})')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)),
                        help='Seed indices to average over (default: 0 1 2 3 4)')
    parser.add_argument('--out', default='pruning_plots.png',
                        help='Base output path; condition + metric suffixes are inserted '
                             'before the extension '
                             '(e.g. pruning_plots.png → '
                             'pruning_plots_ftwoaug_accuracy.png, …)')
    args = parser.parse_args()
    for condition in ('ftwoaug', 'ftwaug'):
        plot_condition(args.ablation_dir, condition, args.seeds, args.out)


if __name__ == '__main__':
    main()
