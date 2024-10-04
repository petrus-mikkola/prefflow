#!/bin/bash
#SBATCH --job-name=ablation_lambda_2
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=18000
#SBATCH --time=50:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py -cd conf/experiment/ablation/onemoon --config-name=ablation_lambda_2 --multirun exp.lambda_dist=mixture_uniform_gaussian exp.mixture_success_prob=0.1,0.25,0.333,0.5,0.666,0.75,1.0 exp.seed=$seed