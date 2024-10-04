#!/bin/bash
#SBATCH --job-name=funnel10d
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=24000
#SBATCH --time=50:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py --config-name=funnel10d --multirun exp.seed=11,12,13,14,15