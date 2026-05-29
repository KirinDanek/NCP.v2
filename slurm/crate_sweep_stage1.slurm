#!/bin/bash
#SBATCH --job-name=crate_sweep1
#SBATCH --output=/n/fs/ncp/NCP.v2/logs/crate_sweep1_%A_%a.out
#SBATCH --error=/n/fs/ncp/NCP.v2/logs/crate_sweep1_%A_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /n/fs/ncp/env1

mkdir -p /n/fs/ncp/NCP.v2/logs
mkdir -p /n/fs/ncp/NCP.v2/results/crate_sweep_stage1

# ---------------------------------------------------------------------------
# 8-job array: iter_finetune_epochs × rank_loader_type
#
#   TASK  iter_finetune_epochs  rank_loader_type
#   0      0                    positive_only
#   1      2                    positive_only
#   2      5                    positive_only
#   3     10                    positive_only
#   4      0                    full_loader
#   5      2                    full_loader
#   6      5                    full_loader
#   7     10                    full_loader
#
# All runs: NCP, crate, spurious_subspace=1, seed=0, 80% pruning.
# Evaluation is on the val set (150/class × 4 subgroups = 600 entries).
# The held-out test set is never loaded or evaluated in this script.
# After selecting the best hyperparams, re-run with --eval_on_test to
# generate paper-ready plots on the test set.
# ---------------------------------------------------------------------------

TASK=${SLURM_ARRAY_TASK_ID}

ITER_EPOCHS=(0 2 5 10 0 2 5 10)
RANK_TYPES=(positive_only positive_only positive_only positive_only \
            full_loader   full_loader   full_loader   full_loader)

ITER_E=${ITER_EPOCHS[$TASK]}
RANK_T=${RANK_TYPES[$TASK]}

OUT_DIR=/n/fs/ncp/NCP.v2/results/crate_sweep_stage1/iter${ITER_E}_${RANK_T}

echo "Task ${TASK}: iter_finetune_epochs=${ITER_E}  rank_loader_type=${RANK_T}"

cd /n/fs/ncp/NCP.v2

python run_watermark_pruning_experiment.py \
    --experiment            crate \
    --pruner                ncp \
    --seed                  0 \
    --spurious_subspace     1 \
    --pr_step               0.05 \
    --total_pr              0.80 \
    --lr                    1e-4 \
    --momentum              0.9 \
    --train_batch_size      32 \
    --test_batch_size       32 \
    --warmup_epochs         15 \
    --final_finetune_epochs 5 \
    --iter_finetune_epochs  ${ITER_E} \
    --rank_loader_type      ${RANK_T} \
    --out_dir               ${OUT_DIR}
