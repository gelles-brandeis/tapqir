#!/bin/bash
#SBATCH --job-name=height
#SBATCH --output=data/height%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-7

heights=(300 500 600 750 1000 1500 2000 3000)

# Path to your executable
python cosmos_simulations.py -it 40000 -infer 7000 \
  --gain 7 --pi 0.15 --lamda 0.15 --proximity 0.2 \
  -bs 4 --height ${heights[${SLURM_ARRAY_TASK_ID}]} \
  --path data/height${heights[${SLURM_ARRAY_TASK_ID}]} --cuda
