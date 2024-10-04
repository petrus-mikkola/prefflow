#!/bin/bash
#SBATCH --job-name=ablation_llm_k
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24000
#SBATCH --time=50:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py --config-name=ablation_llm_k --multirun data.k=2,3,4,5