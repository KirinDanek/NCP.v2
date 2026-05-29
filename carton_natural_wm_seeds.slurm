#!/bin/bash
#SBATCH --job-name=carton_nat_wm
#SBATCH --output=/n/fs/ncp/NCP.v2/logs/carton_nat_wm_%A_%a.out
#SBATCH --error=/n/fs/ncp/NCP.v2/logs/carton_nat_wm_%A_%a.err
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /n/fs/ncp/env1

mkdir -p /n/fs/ncp/NCP.v2/logs
mkdir -p /n/fs/ncp/NCP.v2/results/carton_natural_wm

SEED=${SLURM_ARRAY_TASK_ID}

echo "Seed ${SEED}: carton, ss3, full_loader, iter_finetune_epochs=5, natural test watermarks"

cd /n/fs/ncp/NCP.v2

python run_watermark_pruning_experiment.py \
    --experiment            carton \
    --pruner                ncp \
    --seed                  ${SEED} \
    --spurious_subspace     3 \
    --pr_step               0.05 \
    --total_pr              0.80 \
    --lr                    1e-4 \
    --momentum              0.9 \
    --train_batch_size      32 \
    --test_batch_size       32 \
    --warmup_epochs         15 \
    --final_finetune_epochs 5 \
    --iter_finetune_epochs  5 \
    --rank_loader_type      full_loader \
    --natural_test \
    --test_indices_path     /n/fs/ncp/NCP.v2/data/images/carton_dugong/test-indices.txt \
    --n_natural_test        400 \
    --out_dir               /n/fs/ncp/NCP.v2/results/carton_natural_wm