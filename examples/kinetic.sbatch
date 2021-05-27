#!/bin/bash
#SBATCH --job-name=kinetic
#SBATCH --output=data/kinetic%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-11

kon=(0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.03 0.03 0.03 0.03)
lamda=(0.01 0.15 0.5 1 0.01 0.15 0.5 1 0.01 0.15 0.5 1 0.01 0.15 0.5 1)

# Path to your executable
python kinetic_simulations.py -it 60000 --num-samples 1000 \
  -N 100 -F 1000 -bs 32 --cuda --dtype double \
  --kon ${kon[${SLURM_ARRAY_TASK_ID}]} \
  --lamda ${lamda[${SLURM_ARRAY_TASK_ID}]} \
  --path data/kon${kon[${SLURM_ARRAY_TASK_ID}]}lamda${lamda[${SLURM_ARRAY_TASK_ID}]}
