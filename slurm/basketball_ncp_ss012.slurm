#!/bin/bash
#SBATCH --job-name=bball_ncp_ss012
#SBATCH --output=/n/fs/ncp/NCP.v2/results/basketball-ncp/logs/bball_ncp_ss012_%j.out
#SBATCH --error=/n/fs/ncp/NCP.v2/results/basketball-ncp/logs/bball_ncp_ss012_%j.err
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Ablate subspaces 0, 1, 2 — keeping only subspace 3 (basketball).
# iter_finetune_epochs=2, positive-only rank loader (original config).
# Generates a per-iteration LRP comparison heatmap after every prune+finetune step.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /n/fs/ncp/env1

OUT_DIR=/n/fs/ncp/NCP.v2/results/basketball-ncp/irrelevant_subspaces_012
HEATMAP_DIR=${OUT_DIR}/heatmaps

mkdir -p "$OUT_DIR"
mkdir -p "$HEATMAP_DIR"
mkdir -p /n/fs/ncp/NCP.v2/results/basketball-ncp/logs

echo "Ablating subspaces [0,1,2] — keeping only basketball subspace 3"

cd /n/fs/ncp/NCP.v2

python run_PFT_basketball.py \
    --pruner                    ncp \
    --u_filepath                /n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt \
    --subspace_dims             128 128 128 128 \
    --irrelevant_subspaces      0 1 2 \
    --iter_finetune_epochs      2 \
    --final_finetune_epochs     5 \
    --out_dir                   ${OUT_DIR} \
    --iter_heatmap_dir          ${HEATMAP_DIR} \
    --vanilla_pruned_path       /n/fs/ncp/NCP.v2/results/basketball-vanilla/van.pth \
    --heatmap_img_dir           /n/fs/ncp/NCP.v2/data/images/drsa_basketball_test_images
