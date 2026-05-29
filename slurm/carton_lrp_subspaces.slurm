#!/bin/bash
#SBATCH --job-name=carton_lrp_sub
#SBATCH --output=/n/fs/ncp/NCP.v2/results/carton_lrp_subspaces_%j.out
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00

PYTHON=/n/fs/ncp/env1/bin/python

cd /n/fs/ncp/NCP.v2
$PYTHON carton_crate_lrp_subspaces.py
