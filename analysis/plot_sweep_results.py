"""
plot_sweep_results.py

Load all stage-1 sweep stats.pt files and generate diagnostic plots.

Usage
-----
python plot_sweep_results.py \
    --sweep_dir /n/fs/ncp/NCP.v2/results/crate_sweep_stage1 \
    --out_dir   /n/fs/ncp/NCP.v2/results/crate_sweep_stage1/plots

Expects one stats.pt per config at:
  {sweep_dir}/iter{N}_{rank_type}/crate/ncp/seed0/stats.pt

Plots generated
---------------
1. sweep_recall_gap.png  — WM recall gap (recall_g1 - recall_g0) vs pruning iter,
                           one panel per rank_loader_type, lines per iter_epochs.
2. sweep_acc.png         — overall accuracy vs pruning iter (same layout).
3. sweep_subgroups.png   — per-subgroup accuracy vs pruning iter,
                           2×4 grid (rank_type × iter_epochs).
4. sweep_summary.png     — bar chart of final-iteration recall gap and accuracy
                           for all 8 configs.
"""

import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ITER_EPOCHS   = [0, 2, 5, 10]
RANK_TYPES    = ['positive_only', 'full_loader']
RANK_LABELS   = {'positive_only': 'pos-only (500)',
                 'full_loader':   'full (250+250)'}
COLORS        = {0: '#1f77b4', 2: '#ff7f0e', 5: '#2ca02c', 10: '#d62728'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stats(sweep_dir, iter_e, rank_t):
    p = Path(sweep_dir) / f'iter{iter_e}_{rank_t}' / 'crate' / 'ncp' / 'seed0' / 'stats.pt'
    if not p.exists():
        print(f'  [missing] {p}')
        return None
    d = torch.load(p, map_location='cpu', weights_only=False)
    return d


def extract_subgroup_series(stats_list, key):
    """Extract per-iteration scalar from eval_subgroup_stats list."""
    return [s.get(key, float('nan')) for s in stats_list]


def recall_gap(stats_list):
    """recall_g1 - recall_g0 at each checkpoint."""
    r0 = extract_subgroup_series(stats_list, 'recall_g0')
    r1 = extract_subgroup_series(stats_list, 'recall_g1')
    return [r1i - r0i for r1i, r0i in zip(r1, r0)]


def niters(stats_list):
    return [s.get('niter', i) for i, s in enumerate(stats_list)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_recall_gap(all_data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, rank_t in zip(axes, RANK_TYPES):
        for iter_e in ITER_EPOCHS:
            d = all_data.get((iter_e, rank_t))
            if d is None:
                continue
            sg = d.get('eval_subgroup_stats', [])
            if not sg:
                continue
            iters = niters(sg)
            gap   = recall_gap(sg)
            ax.plot(iters, gap, marker='o', markersize=3,
                    color=COLORS[iter_e], label=f'iter_e={iter_e}')
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_title(RANK_LABELS[rank_t])
        ax.set_xlabel('Pruning iteration')
        ax.set_ylabel('WM recall gap  (recall_g1 − recall_g0)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Stage-1 sweep: WM recall gap vs pruning iter  (crate, NCP, seed=0)')
    fig.tight_layout()
    out = Path(out_dir) / 'sweep_recall_gap.png'
    fig.savefig(out, dpi=150)
    print(f'[saved] {out}')
    plt.close(fig)


def plot_overall_acc(all_data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, rank_t in zip(axes, RANK_TYPES):
        for iter_e in ITER_EPOCHS:
            d = all_data.get((iter_e, rank_t))
            if d is None:
                continue
            iters = d.get('eval_iter', [])
            accs  = d.get('eval_acc',  [])
            if not iters:
                continue
            ax.plot(iters, accs, marker='o', markersize=3,
                    color=COLORS[iter_e], label=f'iter_e={iter_e}')
        ax.set_title(RANK_LABELS[rank_t])
        ax.set_xlabel('Pruning iteration')
        ax.set_ylabel('Val accuracy (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Stage-1 sweep: overall accuracy vs pruning iter  (crate, NCP, seed=0)')
    fig.tight_layout()
    out = Path(out_dir) / 'sweep_acc.png'
    fig.savefig(out, dpi=150)
    print(f'[saved] {out}')
    plt.close(fig)


def plot_subgroups(all_data, out_dir):
    """4 subgroup accuracies per config, 2×4 grid."""
    SUBGROUPS = [
        ('acc_g0y1', 'c1w0 (pos-NWM)'),
        ('acc_g1y1', 'c1w1 (pos-WM)'),
        ('acc_g0y0', 'c0w0 (neg-NWM)'),
        ('acc_g1y0', 'c0w1 (neg-WM)'),
    ]
    SG_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    n_rows = len(RANK_TYPES)
    n_cols = len(ITER_EPOCHS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7), sharey=True, sharex=False)

    for ri, rank_t in enumerate(RANK_TYPES):
        for ci, iter_e in enumerate(ITER_EPOCHS):
            ax = axes[ri][ci]
            d = all_data.get((iter_e, rank_t))
            title = f'iter_e={iter_e}\n{RANK_LABELS[rank_t]}'
            if d is None:
                ax.set_title(title + '\n[missing]')
                continue
            sg = d.get('eval_subgroup_stats', [])
            iters = niters(sg)
            for (key, label), color in zip(SUBGROUPS, SG_COLORS):
                vals = extract_subgroup_series(sg, key)
                ax.plot(iters, vals, marker='.', markersize=2,
                        color=color, label=label)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('iter', fontsize=7)
            ax.grid(alpha=0.3)
            if ci == 0:
                ax.set_ylabel('Accuracy', fontsize=8)
            if ri == 0 and ci == n_cols - 1:
                ax.legend(fontsize=6, loc='lower left')

    fig.suptitle('Stage-1 sweep: per-subgroup accuracy  (crate, NCP, seed=0)')
    fig.tight_layout()
    out = Path(out_dir) / 'sweep_subgroups.png'
    fig.savefig(out, dpi=150)
    print(f'[saved] {out}')
    plt.close(fig)


def plot_summary(all_data, out_dir):
    """Bar chart of final-iteration recall gap and accuracy for all 8 configs."""
    labels, gaps, accs = [], [], []
    for rank_t in RANK_TYPES:
        for iter_e in ITER_EPOCHS:
            d = all_data.get((iter_e, rank_t))
            label = f'e={iter_e}\n{rank_t[:3]}'
            labels.append(label)
            if d is None:
                gaps.append(float('nan'))
                accs.append(float('nan'))
                continue
            sg = d.get('eval_subgroup_stats', [])
            if sg:
                last = sg[-1]
                r0 = last.get('recall_g0', float('nan'))
                r1 = last.get('recall_g1', float('nan'))
                gaps.append(r1 - r0)
            else:
                gaps.append(float('nan'))
            ea = d.get('eval_acc', [])
            accs.append(ea[-1] if ea else float('nan'))

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, gaps, color=['#1f77b4' if 'pos' in l else '#ff7f0e' for l in labels])
    ax1.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Final WM recall gap  (recall_g1 − recall_g0)')
    ax1.set_title('Recall gap at end of pruning\n(lower → better WM removal)')
    ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(x, accs, color=['#1f77b4' if 'pos' in l else '#ff7f0e' for l in labels])
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Final val accuracy (%)')
    ax2.set_title('Overall accuracy at end of pruning')
    ax2.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='positive_only'),
                       Patch(facecolor='#ff7f0e', label='full_loader')]
    ax1.legend(handles=legend_elements, fontsize=8)

    fig.suptitle('Stage-1 sweep summary  (crate, NCP, seed=0, final pruning iter)')
    fig.tight_layout()
    out = Path(out_dir) / 'sweep_summary.png'
    fig.savefig(out, dpi=150)
    print(f'[saved] {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir', type=str,
                   default='/n/fs/ncp/NCP.v2/results/crate_sweep_stage1')
    p.add_argument('--out_dir',   type=str, default=None,
                   help='Output directory for plots. Defaults to sweep_dir/plots.')
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.sweep_dir) / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all configs
    all_data = {}
    for rank_t in RANK_TYPES:
        for iter_e in ITER_EPOCHS:
            d = load_stats(args.sweep_dir, iter_e, rank_t)
            all_data[(iter_e, rank_t)] = d

    n_loaded = sum(1 for v in all_data.values() if v is not None)
    print(f'Loaded {n_loaded}/8 configs from {args.sweep_dir}')

    if n_loaded == 0:
        print('No data found — exiting.')
        return

    plot_recall_gap(all_data, out_dir)
    plot_overall_acc(all_data, out_dir)
    plot_subgroups(all_data, out_dir)
    plot_summary(all_data, out_dir)
    print(f'\nAll plots saved to {out_dir}')


if __name__ == '__main__':
    main()
