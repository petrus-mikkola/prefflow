#!/bin/bash
#SBATCH --job-name=ablation_k_n_funnel
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24000
#SBATCH --time=50:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python main.py -cd conf/experiment/ablation/funnel --config-name=ablation_k_n --multirun data.k=2,3,5,10 data.n=10000 params.nflows=8 exp.seed=$seed