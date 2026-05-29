#!/bin/sh
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --gres=gpu:0         # the number of GPUs requested
#SBATCH --mem=16G             # memory 
#SBATCH -o outfile            # send stdout to outfile
#SBATCH -e errfile            # send stderr to errfile
#SBATCH -t 01:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=kd9132@princeton.edu

conda init bash
source ~/.bashrc
conda activate /u/kd9132/n/fs/ncp/env1/

python /u/kd9132/n/fs/ncp/NCP.v2/data/images/download_binary_dataset.py
