#!/bin/bash
#SBATCH --job-name=noise_ablation
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=16000
#SBATCH --time=32:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py --config-name=noise_ablation --multirun exp.true_s=1.0,0.01,5 modelparams.s=1.0,0.01,5