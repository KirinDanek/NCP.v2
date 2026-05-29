#!/bin/bash
#SBATCH --job-name=bball_lrp_gamma
#SBATCH --output=/n/fs/ncp/NCP.v2/results/basketball-ncp/logs/basketball_lrp_gamma_%j.out
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:05:00

PYTHON=/n/fs/ncp/env1/bin/python

cd /n/fs/ncp/NCP.v2
$PYTHON basketball_lrp_comparison_gamma.py
