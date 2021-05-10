#!/bin/bash
#SBATCH --job-name=negative
#SBATCH --output=data/negative%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=1-4

lamdas=(0.01 0.05 0.15 0.5 1)

# Path to your executable
python cosmos_simulations.py -it 50000 -infer 50000 \
  --gain 7 --pi 0 --height 3000 --proximity 0.2 \
  -bs 4 --lamda ${lamdas[${SLURM_ARRAY_TASK_ID}]} \
  --path data/negative${lamdas[${SLURM_ARRAY_TASK_ID}]} --cuda
