#!/bin/bash
#SBATCH --mem=80G
#SBATCH --time=90:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=dcs-res
#SBATCH --cpus-per-task=4
#SBATCH --partition=dcs-gpu

nvidia-smi

module load Anaconda3/2019.07
source activate torch2

torchrun --nnodes=1 --nproc_per_node=1 --standalone train_ddp.py -c ./configs/mamba_test_bessemer.yaml -ws 1 -dtype f16

echo "WE ARE DONE, BYE"



#// --gpus-per-node=1SBATCH --account=dcs-res SBATCH --partition=dcs-gpu
###SBATCH --gpus-per-node=1
###SBATCH --account=dcs-res
###SBATCH --partition=dcs-gpu
