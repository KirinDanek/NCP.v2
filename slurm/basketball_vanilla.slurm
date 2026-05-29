#!/bin/bash
#SBATCH --job-name=basketball_vanilla
#SBATCH --output=/n/fs/ncp/NCP.v2/results/basketball-vanilla/logs/basketball_vanilla_%j.out
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Vanilla LRP pruner — no NCP subspace projection.
# Parameters mirror basketball_ncp.slurm (same pr_step, total_pr, rank_n_images,
# warmup settings, etc.) except --pruner vanilla and no --u_filepath /
# --irrelevant_subspaces arguments.
# Saved checkpoint: results/basketball-vanilla/van.pth

PYTHON=/n/fs/ncp/env1/bin/python

OUT_DIR="/n/fs/ncp/NCP.v2/results/basketball-vanilla"
mkdir -p "$OUT_DIR/logs"

echo "Job ${SLURM_JOB_ID}: vanilla LRP pruning"

cd /n/fs/ncp/NCP.v2
$PYTHON run_PFT_basketball.py \
    --pruner vanilla \
    --out_dir "$OUT_DIR"
