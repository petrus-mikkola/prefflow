#!/bin/bash
#SBATCH --job-name=llm-prior
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12000
#SBATCH --time=50:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py --config-name=llm --multirun modelparams.weightprior=1.0,0.0