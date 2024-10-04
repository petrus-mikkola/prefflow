#!/bin/bash
#SBATCH --job-name=onemoon
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16000
#SBATCH --time=60:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py --config-name=onemoon --multirun exp.seed=11,12,13,14,15,16,17,18,19,20