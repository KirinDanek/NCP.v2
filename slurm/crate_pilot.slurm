#!/bin/bash
#SBATCH --job-name=crate_pilot
#SBATCH --output=/n/fs/ncp/NCP.v2/logs/crate_pilot_%A_%a.out
#SBATCH --error=/n/fs/ncp/NCP.v2/logs/crate_pilot_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /n/fs/ncp/env1

mkdir -p /n/fs/ncp/NCP.v2/logs
mkdir -p /n/fs/ncp/NCP.v2/results/crate_pilot

# ---------------------------------------------------------------------------
# 4-job array:
#   0 — diagnostic only (warmup + pre-pruning ablation eval)
#   1 — NCP, subspace 1, 2 iter epochs  (matches the full run)
#   2 — NCP, subspace 3, 2 iter epochs  (candidate fix)
#   3 — NCP, subspace 1, 10 iter epochs (matches old pipeline)
# ---------------------------------------------------------------------------

cd /n/fs/ncp/NCP.v2

TASK=${SLURM_ARRAY_TASK_ID}

if [ $TASK -eq 0 ]; then
    echo "Task 0: pre-pruning ablation diagnostic"
    python crate_ablation_diagnostic.py \
        --seed 0 \
        --warmup_epochs 15 \
        --lr 1e-4 \
        --batch_size 32 \
        --out /n/fs/ncp/NCP.v2/results/crate_pilot/ablation_diagnostic.pt

elif [ $TASK -eq 1 ]; then
    echo "Task 1: NCP pilot — subspace 1, 2 iter epochs"
    python run_watermark_pruning_experiment.py \
        --experiment   crate \
        --pruner       ncp \
        --seed         0 \
        --spurious_subspace 1 \
        --pr_step      0.05 \
        --total_pr     0.80 \
        --lr           1e-4 \
        --momentum     0.9 \
        --train_batch_size 32 \
        --test_batch_size  32 \
        --warmup_epochs    15 \
        --final_finetune_epochs 5 \
        --iter_finetune_epochs  2 \
        --out_dir /n/fs/ncp/NCP.v2/results/crate_pilot/ss1_e2/

elif [ $TASK -eq 2 ]; then
    echo "Task 2: NCP pilot — subspace 3, 2 iter epochs"
    python run_watermark_pruning_experiment.py \
        --experiment   crate \
        --pruner       ncp \
        --seed         0 \
        --spurious_subspace 3 \
        --pr_step      0.05 \
        --total_pr     0.80 \
        --lr           1e-4 \
        --momentum     0.9 \
        --train_batch_size 32 \
        --test_batch_size  32 \
        --warmup_epochs    15 \
        --final_finetune_epochs 5 \
        --iter_finetune_epochs  2 \
        --out_dir /n/fs/ncp/NCP.v2/results/crate_pilot/ss3_e2/

elif [ $TASK -eq 3 ]; then
    echo "Task 3: NCP pilot — subspace 1, 10 iter epochs (old pipeline)"
    python run_watermark_pruning_experiment.py \
        --experiment   crate \
        --pruner       ncp \
        --seed         0 \
        --spurious_subspace 1 \
        --pr_step      0.05 \
        --total_pr     0.80 \
        --lr           1e-4 \
        --momentum     0.9 \
        --train_batch_size 32 \
        --test_batch_size  32 \
        --warmup_epochs    15 \
        --final_finetune_epochs 5 \
        --iter_finetune_epochs  10 \
        --out_dir /n/fs/ncp/NCP.v2/results/crate_pilot/ss1_e10/
fi
