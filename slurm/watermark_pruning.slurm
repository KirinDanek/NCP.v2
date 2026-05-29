#!/bin/bash
#SBATCH --job-name=watermark_pruning
#SBATCH --output=/n/fs/ncp/NCP.v2/logs/watermark_pruning_%A_%a.out
#SBATCH --error=/n/fs/ncp/NCP.v2/logs/watermark_pruning_%A_%a.err
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /n/fs/ncp/env1

mkdir -p /n/fs/ncp/NCP.v2/logs
mkdir -p /n/fs/ncp/NCP.v2/results/watermark_experiment

# ---------------------------------------------------------------------------
# Watermark spurious-cue NCP experiment
# 20 jobs: 2 experiments × 2 pruners × 5 seeds
#
# Job index layout:
#   0– 4 : carton / ncp     / seeds 0–4
#   5– 9 : carton / vanilla / seeds 0–4
#  10–14 : crate  / ncp     / seeds 0–4
#  15–19 : crate  / vanilla / seeds 0–4
# ---------------------------------------------------------------------------

# ── Set the DRSA subspace index that captures the hanzi watermark ──────────
# Inspect carton_lrp_subspaces.png and crate_lrp_subspaces.png to identify
# which subspace (0-3) corresponds to the watermark feature, then set below.
CARTON_SPURIOUS_SUBSPACE=3    # <-- update after subspace analysis
CRATE_SPURIOUS_SUBSPACE=1     # <-- update after subspace analysis
# ──────────────────────────────────────────────────────────────────────────

TASK=${SLURM_ARRAY_TASK_ID}

# Decode experiment / pruner / seed from task index
EXP_IDX=$(( TASK / 10 ))         # 0=carton, 1=crate
WITHIN=$(( TASK % 10 ))
PRUNER_IDX=$(( WITHIN / 5 ))     # 0=ncp,    1=vanilla
SEED=$(( WITHIN % 5 ))           # 0..4

if [ $EXP_IDX -eq 0 ]; then
    EXPERIMENT=carton
    SPURIOUS_SUBSPACE=${CARTON_SPURIOUS_SUBSPACE}
else
    EXPERIMENT=crate
    SPURIOUS_SUBSPACE=${CRATE_SPURIOUS_SUBSPACE}
fi

if [ $PRUNER_IDX -eq 0 ]; then
    PRUNER=ncp
else
    PRUNER=vanilla
fi

echo "Task ${TASK}: experiment=${EXPERIMENT}  pruner=${PRUNER}  seed=${SEED}  spurious_subspace=${SPURIOUS_SUBSPACE}"

mkdir -p /n/fs/ncp/NCP.v2/logs

cd /n/fs/ncp/NCP.v2

python run_watermark_pruning_experiment.py \
    --experiment   ${EXPERIMENT} \
    --pruner       ${PRUNER} \
    --seed         ${SEED} \
    --spurious_subspace ${SPURIOUS_SUBSPACE} \
    --pr_step      0.05 \
    --total_pr     0.80 \
    --lr           1e-4 \
    --momentum     0.9 \
    --train_batch_size 32 \
    --test_batch_size  32 \
    --warmup_epochs    15 \
    --final_finetune_epochs 5 \
    --out_dir /n/fs/ncp/NCP.v2/results/watermark_experiment/
