#!/bin/bash
#SBATCH --job-name=ablation_k_n
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20000
#SBATCH --time=48:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py -cd conf/experiment/ablation/gaussian --config-name=ablation_k_n --multirun data.k=2,3,5,10 data.n=1000 params.nflows=6 exp.seed=$seed