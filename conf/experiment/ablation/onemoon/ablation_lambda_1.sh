#!/bin/bash
#SBATCH --job-name=ablation_lambda_1
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=16000
#SBATCH --time=60:00:00

# set the number of threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py -cd conf/experiment/ablation/onemoon --config-name=ablation_lambda_1 --multirun exp.lambda_dist=mixture_uniform_gaussian,target,uniform exp.seed=$seed