#!/bin/sh
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1          # the number of GPUs requested
#SBATCH --mem=32G             # memory 
#SBATCH -o outfile           # send stdout to outfile
#SBATCH -e errfile           # send stderr to errfile
#SBATCH -t 02:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=kd9132@princeton.edu

conda init bash
source ~/.bashrc
conda activate /n/fs/ncp/env1/

python /n/fs/ncp/NCP.v2/data/train_vgg.py
