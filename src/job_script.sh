#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
module load python
srun ./predinas

