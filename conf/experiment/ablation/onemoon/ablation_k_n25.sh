#!/bin/bash
#SBATCH --job-name=ablation_k_n
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18000
#SBATCH --time=48:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py -cd conf/experiment/ablation/onemoon --config-name=ablation_k_n --multirun data.k=2,3,5,10 data.n=25 params.nflows=12 exp.seed=$seed