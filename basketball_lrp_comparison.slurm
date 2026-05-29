#!/bin/bash
#SBATCH --job-name=basketball_lrp_cmp
#SBATCH --output=/n/fs/ncp/NCP.v2/results/basketball-ncp/logs/basketball_lrp_cmp_%j.out
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

PYTHON=/n/fs/ncp/env1/bin/python

cd /n/fs/ncp/NCP.v2
$PYTHON -c "
import basketball_lrp_comparison as m
m.OUT_PATH = m.RESULT_DIR / 'lrp_comparison_subspace3_with_vanilla.png'
m.main()
"
