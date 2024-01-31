#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e error_%A_%a.out
#SBATCH --mem-per-cpu=5GB

module load anaconda3/2022.05-x86_64
source activate resenv
python ensemble_lorenz96.py $SLURM_ARRAY_TASK_ID
