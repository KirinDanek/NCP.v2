#!/bin/bash
#SBATCH --job-name=wm_smoke_test
#SBATCH --output=/n/fs/ncp/NCP.v2/logs/wm_smoke_test_%j.out
#SBATCH --error=/n/fs/ncp/NCP.v2/logs/wm_smoke_test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /n/fs/ncp/env1

mkdir -p /n/fs/ncp/NCP.v2/logs
mkdir -p /n/fs/ncp/NCP.v2/results/watermark_experiment

cd /n/fs/ncp/NCP.v2

# Smoke-test: carton, both pruners, seed 0.
# Uses --total_pr 0.10 (~2 iterations) and minimal warmup/finetune
# to quickly exercise the full pipeline without a full run.
SPURIOUS=3   # carton spurious subspace

for PRUNER in ncp vanilla; do
    echo "=== Smoke test: carton / ${PRUNER} / seed 0 ==="
    python run_watermark_pruning_experiment.py \
        --experiment       carton \
        --pruner           ${PRUNER} \
        --seed             0 \
        --spurious_subspace ${SPURIOUS} \
        --total_pr         0.10 \
        --pr_step          0.05 \
        --warmup_epochs    2 \
        --final_finetune_epochs 1 \
        --out_dir /n/fs/ncp/NCP.v2/results/watermark_experiment_smoke/
    echo "=== Done: carton / ${PRUNER} / seed 0 ==="
done

echo "Smoke test complete. Check results in results/watermark_experiment_smoke/"
