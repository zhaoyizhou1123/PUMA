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

# Define your parameters
SEEDS=(2026 2027 2028)
START_PORT=29501

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
PORT=$((START_PORT + SLURM_ARRAY_TASK_ID))

export PYTHONUNBUFFERED=1
MASTER_ADDR=localhost

echo "Running SEED: $SEED | PORT: $PORT"

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$PORT \
  -m maze.train_proseco_corrected --config-path "../yaml_files/sudoku" --config-name base \
    training.strategy=proseco \
    training.eval_steps=5000 \
    training.save_steps=5000 \
    +training.ckpt_root="/projects/bgqz/zzhou24/checkpoints/" \
    validation.val_dir=/projects/bgqz/zzhou24/data/sudoku \
    validation.sampling.unmasking_num=[1,3,9] \
    +validation.sampling.edit_freq=[3] \
    +validation.sampling.edit_step=[4] \
    validation.track=False \
    data.seed=$SEED
