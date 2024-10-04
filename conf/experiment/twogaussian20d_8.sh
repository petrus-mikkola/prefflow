#!/bin/bash
#SBATCH --job-name=twogaussians20d
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000
#SBATCH --time=70:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py --config-name=twogaussian20d --multirun exp.seed=20