#!/bin/bash
#SBATCH --job-name=seed
#SBATCH --output=data/seed%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-16

# Path to your executable
python cosmos_simulations.py --seed ${SLURM_ARRAY_TASK_ID} \
  -it 40000 -infer 5000 -bs 4 --path data/seed${SLURM_ARRAY_TASK_ID} --cuda
