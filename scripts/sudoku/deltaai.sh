#!/bin/bash
#SBATCH --job-name=sudoku
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=5:00:00
#SBATCH --output=slurm/%x/job_%j_%a.out
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh
#SBATCH --array=0-2

#SBATCH --chdir=/u/zzhou24/projects/PUMA

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load cuda/12.6.1
source ~/miniconda3/bin/activate
conda activate puma

dir=scripts/sudoku
main=proseco.sh

# Define your parameters
SEEDS=(2026 2027 2028)
START_PORT=29501

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
PORT=$((START_PORT + SLURM_ARRAY_TASK_ID))

bash $dir/$main ${SEED} ${PORT}
