#!/bin/bash
#SBATCH --job-name=lips_viz
#SBATCH --output=/n/fs/ncp/NCP.v2/results/lips_viz_%j.out
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

PYTHON=/n/fs/ncp/env1/bin/python

mkdir -p /n/fs/ncp/NCP.v2/results

echo "Job ${SLURM_JOB_ID}: lips subspace montage visualization"

cd /n/fs/ncp/NCP.v2
$PYTHON visualize_lips_subspace.py
