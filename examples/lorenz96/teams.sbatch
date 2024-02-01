#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p sched_mit_pog_r8
#SBATCH -e error_%A_%a.out
#SBATCH --nodelist=node2421
#SBATCH --mem-per-cpu=10GB

module load anaconda3/2022.05-x86_64
source activate resenv
python -u teams_manager_lorenz96.py 0 $SLURM_ARRAY_TASK_ID {0..55} > progress_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".out
