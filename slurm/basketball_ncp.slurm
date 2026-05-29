#!/bin/bash
#SBATCH --job-name=basketball_ncp
#SBATCH --output=/n/fs/ncp/NCP.v2/results/basketball-ncp/logs/basketball_ncp_%A_%a.out
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-2

# Array layout: 3 jobs, one per irrelevant subspace index
#   task 0 → --irrelevant_subspaces 1
#   task 1 → --irrelevant_subspaces 2
#   task 2 → --irrelevant_subspaces 3
#
# U_basketball_tensor.pt is 512×512; split into 4 subspaces of 128 each.
# Indices are 0-based: 0, 1, 2, 3.

PYTHON=/n/fs/ncp/env1/bin/python
IRRELEVANT_SUBSPACE=$(( SLURM_ARRAY_TASK_ID + 1 ))  # maps 0→1, 1→2, 2→3

U_FILEPATH=/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt
SUBSPACE_DIMS="128 128 128 128"

OUT_DIR="/n/fs/ncp/NCP.v2/results/basketball-ncp/irrelevant_subspace_${IRRELEVANT_SUBSPACE}"
mkdir -p "$OUT_DIR"

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: irrelevant_subspaces=${IRRELEVANT_SUBSPACE}"

cd /n/fs/ncp/NCP.v2
$PYTHON run_PFT_basketball.py \
    --pruner ncp \
    --u_filepath "$U_FILEPATH" \
    --subspace_dims $SUBSPACE_DIMS \
    --irrelevant_subspaces $IRRELEVANT_SUBSPACES \
    --out_dir "$OUT_DIR"
