#!/usr/bin/env python3
"""
plot_watermark_results.py

Plots NCP vs vanilla pruning results for the watermark spurious-cue experiment.
Produces four figures per experiment (carton / crate):

  {experiment}_accuracy.png        — overall test accuracy vs pruning iteration
  {experiment}_subgroup_acc.png    — per-subgroup accuracy (c1w0, c1w1, c0w0, c0w1)
  {experiment}_recall.png          — per-watermark-group recall + |gap|
  {experiment}_precision.png       — per-watermark-group precision + |gap|

Each line is the mean over --seeds; shaded band is ±1σ.

Subgroup key:
  c1w0  acc_g0y1  positive class, no watermark   (hardest for model without spurious cue)
  c1w1  acc_g1y1  positive class, watermarked     (model can use spurious cue)
  c0w0  acc_g0y0  negative class, no watermark
  c0w1  acc_g1y0  negative class, watermarked     (spurious cue → false positives)

Usage:
    python plot_watermark_results.py \\
        --results_dir /n/fs/ncp/NCP.v2/results/watermark_experiment/ \\
        --experiments carton crate \\
        --seeds 0 1 2 3 4 \\
        --out_dir /n/fs/ncp/NCP.v2/results/watermark_experiment/plots/
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PRUNER_COLORS = {'ncp': 'steelblue', 'vanilla': 'crimson'}

# Watermark group line styles (g=0 no-WM solid, g=1 WM dashed)
WM_STYLE = {0: '-', 1: '--'}
WM_LABEL = {0: 'no-watermark', 1: 'watermark'}

# Subgroup styles for the 4-line subgroup accuracy plot
SUBGROUP_STYLE = {
    ('g0y1', 'c1w0'): ('steelblue',  '-',  'c1w0  pos, no-WM'),
    ('g1y1', 'c1w1'): ('steelblue',  '--', 'c1w1  pos, WM'),
    ('g0y0', 'c0w0'): ('darkorange', '-',  'c0w0  neg, no-WM'),
    ('g1y0', 'c0w1'): ('darkorange', '--', 'c0w1  neg, WM'),
}

MS         = 4
ALPHA_BAND = 0.18
PR_STEP        = 0.05                          # fraction of filters pruned per iteration
MAX_PRUNE_ITER = int(round(0.80 / PR_STEP))   # = 16; iter 17 is final fine-tune, not pruning

# Font sizes for paper figures
FS_LABEL  = 13   # axis labels
FS_TICK   = 11   # tick labels
FS_LEGEND = 11   # legend text


# ---------------------------------------------------------------------------
# Data helpers  (same logic as plot_results.py)
# ---------------------------------------------------------------------------

def _load(path):
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location='cpu', weights_only=False)
    # Normalise watermark-sweep key names (eval_*) to the canonical names
    # used throughout this script (subgroup_stats / test_acc / test_iter).
    if d is not None and 'eval_subgroup_stats' in d and 'subgroup_stats' not in d:
        d = dict(d)
        d['subgroup_stats'] = d['eval_subgroup_stats']
        d['test_acc']       = d.get('eval_acc',  [])
        d['test_iter']      = d.get('eval_iter', [])
    return d


def _acc_metric(d):
    return {int(n): float(v)
            for n, v in zip(d.get('test_iter', []), d.get('test_acc', []))
            if int(n) <= MAX_PRUNE_ITER}


def _sg_metric(d, key):
    return {s['niter']: s.get(key, float('nan'))
            for s in d.get('subgroup_stats', [])
            if s['niter'] <= MAX_PRUNE_ITER}


def _agg(dicts):
    all_niters = sorted({n for d in dicts for n in d})
    means, stds = [], []
    for n in all_niters:
        vals = [d[n] for d in dicts
                if n in d and not math.isnan(float(d[n]))]
        means.append(float(np.mean(vals)) if vals else float('nan'))
        stds.append(float(np.std(vals))   if vals else float('nan'))
    return all_niters, means, stds


def _scale100(d):
    return {n: v * 100 for n, v in d.items()}


def _niters_to_pct(niters):
    """Convert pruning iteration indices to % filters pruned (capped at 80%)."""
    return [min(int(round(n * PR_STEP * 100)), 80) for n in niters]


def _load_seeds(results_dir, experiment, pruner, seeds):
    """Return list of loaded stat dicts (skipping missing files)."""
    ds = []
    for s in seeds:
        path = Path(results_dir) / experiment / pruner / f'seed{s}' / 'stats.pt'
        d = _load(str(path))
        if d is not None:
            ds.append(d)
        else:
            print(f'  [missing] {path}')
    return ds


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_ms(ax, niters, means, stds, color, ls, label):
    pts = [(n, m, s) for n, m, s in zip(niters, means, stds)
           if not math.isnan(m)]
    if not pts:
        return
    ns, ms, ss = zip(*pts)
    ax.plot(list(ns), list(ms), color=color, ls=ls,
            marker='o', ms=MS, label=label)
    ax.fill_between(list(ns),
                    [m - s for m, s in zip(ms, ss)],
                    [m + s for m, s in zip(ms, ss)],
                    color=color, alpha=ALPHA_BAND)


def _style_ax(ax, xlabel='Filters pruned (%)', ylabel=None):
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(left=0, right=83)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.xaxis.set_tick_params(labelsize=FS_TICK)


def _finish(fig, suptitle, out_path):
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


# ---------------------------------------------------------------------------
# Per-figure plot functions
# ---------------------------------------------------------------------------

def plot_accuracy(experiment, ncp_ds, van_ds, n_seeds, out_path, no_titles=False):
    """Overall test accuracy: one line per pruner."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, ds in [('CNP', 'steelblue', ncp_ds),
                              ('Vanilla', 'crimson',  van_ds)]:
        if not ds:
            continue
        niters, means, stds = _agg([_acc_metric(d) for d in ds])
        _plot_ms(ax, _niters_to_pct(niters), means, stds, color=color, ls='-', label=label)
    if not no_titles:
        ax.set_title(f'{experiment.capitalize()} – Test Accuracy vs Filters Pruned')
    _style_ax(ax, ylabel='Test accuracy (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND)
    suptitle = '' if no_titles else f'{experiment.capitalize()} watermark experiment  (n={n_seeds} seeds)'
    _finish(fig, suptitle, out_path)


def plot_subgroup_accuracy(experiment, ncp_ds, van_ds, n_seeds, out_path):
    """
    4-subgroup accuracy panel: two columns (NCP | Vanilla), four lines each.
    Key subgroups:
      c1w0 (acc_g0y1) — positive, no watermark  (tests model without spurious cue)
      c0w1 (acc_g1y0) — negative, watermarked   (tests false-positive from spurious cue)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, label, ds in zip(axes, ['CNP', 'Vanilla'], [ncp_ds, van_ds]):
        if not ds:
            ax.set_visible(False)
            continue
        for (sg_key, sg_name), (color, ls, sg_label) in SUBGROUP_STYLE.items():
            stat_key = f'acc_{sg_key}'
            dicts = [_scale100(_sg_metric(d, stat_key)) for d in ds]
            niters, means, stds = _agg(dicts)
            _plot_ms(ax, _niters_to_pct(niters), means, stds, color=color, ls=ls, label=sg_label)
        ax.set_title(label, fontsize=FS_LABEL)
        _style_ax(ax, ylabel='Accuracy (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND)

    _finish(fig,
            f'{experiment.capitalize()} – Per-Subgroup Accuracy  (n={n_seeds} seeds)',
            out_path)


def _plot_2panel(experiment, ncp_ds, van_ds, n_seeds, key_base,
                 title_main, title_gap, ylabel_main, ylabel_gap, out_path):
    """Per-watermark-group metric (two lines) + |gap| panel — for recall / precision."""
    fig, (ax_main, ax_gap) = plt.subplots(1, 2, figsize=(13, 5))

    for label, color, ds in [('CNP', 'steelblue', ncp_ds),
                              ('Vanilla', 'crimson',  van_ds)]:
        if not ds:
            continue
        for g in (0, 1):
            dicts_g = [_scale100(_sg_metric(d, f'{key_base}_g{g}')) for d in ds]
            niters, means, stds = _agg(dicts_g)
            _plot_ms(ax_main, _niters_to_pct(niters), means, stds,
                     color=color, ls=WM_STYLE[g],
                     label=f'{label} – {WM_LABEL[g]}')

        # Gap per seed, then aggregate
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
        _plot_ms(ax_gap, _niters_to_pct(niters), means, stds, color=color, ls='-', label=label)

    ax_main.set_title(title_main)
    _style_ax(ax_main, ylabel=ylabel_main)
    ax_main.set_ylim(0, 105)

    ax_gap.set_title(title_gap)
    _style_ax(ax_gap, ylabel=ylabel_gap)
    ax_gap.set_ylim(bottom=0)

    # Watermark-group legend entries
    nwm_line = mlines.Line2D([], [], color='grey', ls='-',  label='── no watermark')
    wm_line  = mlines.Line2D([], [], color='grey', ls='--', label='-- watermark')
    h, l = ax_main.get_legend_handles_labels()
    ax_main.legend(h + [nwm_line, wm_line], l + ['── no watermark', '-- watermark'],
                   fontsize=FS_LEGEND - 2)
    ax_gap.legend(fontsize=FS_LEGEND)

    for ax in (ax_main, ax_gap):
        ax.grid(True, alpha=0.3)

    _finish(fig,
            f'{experiment.capitalize()} – {title_main}  (n={n_seeds} seeds)',
            out_path)


# ---------------------------------------------------------------------------
# c0w1-only combined figure (NCP + Vanilla in one panel)
# ---------------------------------------------------------------------------

def plot_c0w1_combined(experiment, ncp_ds, van_ds, n_seeds, out_path, no_titles=False):
    """Single panel: c0w1 (neg, WM) accuracy for NCP and Vanilla on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, color, ds in [('CNP', 'steelblue', ncp_ds),
                              ('Vanilla', 'crimson', van_ds)]:
        if not ds:
            continue
        dicts = [_scale100(_sg_metric(d, 'acc_g1y0')) for d in ds]
        niters, means, stds = _agg(dicts)
        _plot_ms(ax, _niters_to_pct(niters), means, stds, color=color, ls='-', label=label)

    if not no_titles:
        ax.set_title(f'{experiment.capitalize()} – c0w1 accuracy (negative class, watermarked)')
    _style_ax(ax, ylabel='Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND)
    suptitle = '' if no_titles else f'{experiment.capitalize()} watermark experiment  (n={n_seeds} seeds)'
    _finish(fig, suptitle, out_path)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def plot_experiment(results_dir, experiment, seeds, out_dir):
    ncp_ds = _load_seeds(results_dir, experiment, 'ncp',     seeds)
    van_ds = _load_seeds(results_dir, experiment, 'vanilla', seeds)
    n      = max(len(ncp_ds), len(van_ds))

    if n == 0:
        print(f'[{experiment}] No results found — skipping.')
        return

    print(f'[{experiment}] ncp={len(ncp_ds)} seeds  vanilla={len(van_ds)} seeds')

    def out(suffix):
        return str(Path(out_dir) / f'{experiment}_{suffix}.png')

    plot_accuracy(experiment, ncp_ds, van_ds, n, out('accuracy'))
    plot_subgroup_accuracy(experiment, ncp_ds, van_ds, n, out('subgroup_acc'))
    _plot_2panel(experiment, ncp_ds, van_ds, n,
                 key_base='recall',
                 title_main='Per-WM-Group Recall vs Pruning Iteration',
                 title_gap='|Recall gap| (WM − no-WM) vs Pruning Iteration',
                 ylabel_main='Recall (%)', ylabel_gap='|Recall gap| (pp)',
                 out_path=out('recall'))
    _plot_2panel(experiment, ncp_ds, van_ds, n,
                 key_base='precision',
                 title_main='Per-WM-Group Precision vs Pruning Iteration',
                 title_gap='|Precision gap| (WM − no-WM) vs Pruning Iteration',
                 ylabel_main='Precision (%)', ylabel_gap='|Precision gap| (pp)',
                 out_path=out('precision'))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_dir', default='results/watermark_experiment/',
                   help='Root dir containing {experiment}/{pruner}/seed{s}/stats.pt')
    p.add_argument('--experiments', nargs='+', default=['carton', 'crate'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--out_dir', default=None,
                   help='Output directory for PNGs (default: results_dir/plots/)')
    p.add_argument('--c0w1_only', action='store_true', default=False,
                   help='Generate only the combined c0w1 figure (saves as '
                        '{experiment}_c0w1_combined.png); does not overwrite other plots.')
    p.add_argument('--no_titles', action='store_true', default=False,
                   help='Suppress all axes and figure titles (for paper figures). '
                        'Only affects accuracy and c0w1_combined plots when combined '
                        'with the relevant flags.')
    args = p.parse_args()

    out_dir = args.out_dir or str(Path(args.results_dir) / 'plots')

    for exp in args.experiments:
        ncp_ds = _load_seeds(args.results_dir, exp, 'ncp',     args.seeds)
        van_ds = _load_seeds(args.results_dir, exp, 'vanilla', args.seeds)
        n      = max(len(ncp_ds), len(van_ds))
        if n == 0:
            print(f'[{exp}] No results found — skipping.')
            continue

        if args.c0w1_only:
            out_path = str(Path(out_dir) / f'{exp}_c0w1_combined.png')
            plot_c0w1_combined(exp, ncp_ds, van_ds, n, out_path,
                               no_titles=args.no_titles)
        else:
            def out(suffix):
                return str(Path(out_dir) / f'{exp}_{suffix}.png')
            print(f'[{exp}] ncp={len(ncp_ds)} seeds  vanilla={len(van_ds)} seeds')
            plot_accuracy(exp, ncp_ds, van_ds, n, out('accuracy'),
                          no_titles=args.no_titles)
            plot_subgroup_accuracy(exp, ncp_ds, van_ds, n, out('subgroup_acc'))
            _plot_2panel(exp, ncp_ds, van_ds, n,
                         key_base='recall',
                         title_main='Per-WM-Group Recall vs Pruning Iteration',
                         title_gap='|Recall gap| (WM − no-WM) vs Pruning Iteration',
                         ylabel_main='Recall (%)', ylabel_gap='|Recall gap| (pp)',
                         out_path=out('recall'))
            _plot_2panel(exp, ncp_ds, van_ds, n,
                         key_base='precision',
                         title_main='Per-WM-Group Precision vs Pruning Iteration',
                         title_gap='|Precision gap| (WM − no-WM) vs Pruning Iteration',
                         ylabel_main='Precision (%)', ylabel_gap='|Precision gap| (pp)',
                         out_path=out('precision'))


if __name__ == '__main__':
    main()
