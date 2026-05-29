#!/bin/bash
#SBATCH --job-name=bball_lrp_simple
#SBATCH --output=/n/fs/ncp/NCP.v2/results/basketball-ncp/logs/basketball_lrp_simple_%j.out
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

PYTHON=/n/fs/ncp/env1/bin/python

cd /n/fs/ncp/NCP.v2
$PYTHON basketball_lrp_comparison_simple.py
