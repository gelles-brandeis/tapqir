#!/bin/bash
#SBATCH --job-name=seed
#SBATCH --output=data/seed%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-16

# Path to your executable
python cosmos_simulations.py -it 60000 --num-samples 1000 \
  -bs 4 --cuda --dtype double \
  --seed ${SLURM_ARRAY_TASK_ID} \
  --path data/seed${SLURM_ARRAY_TASK_ID}
