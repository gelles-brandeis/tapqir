#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=data5/analysis%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-11

kon=(0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.03 0.03 0.03 0.03)
lamda=(0.01 0.15 0.5 1 0.01 0.15 0.5 1 0.01 0.15 0.5 1 0.01 0.15 0.5 1)

# Path to your executable
python kinetic_analysis.py -it 1000 \
  --data-path data5/kon${kon[${SLURM_ARRAY_TASK_ID}]}lamda${lamda[${SLURM_ARRAY_TASK_ID}]} \
  --param-path data5/kon${kon[${SLURM_ARRAY_TASK_ID}]}lamda${lamda[${SLURM_ARRAY_TASK_ID}]}/runs/cosmos/v1.1.6/S1/control/lr0.005/bs32
