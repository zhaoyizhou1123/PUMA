#!/bin/bash
#SBATCH --job-name=puma_backplay_hard_debug
#SBATCH --partition=general
#SBATCH --output=logs/train_sudoku_backplay_hard_debug_%j.log
#SBATCH --error=logs/train_sudoku_backplay_hard_debug_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32GB
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-20

# Debug run for BackPlay fine-tuning on sudoku_hard.
# 50 steps, no wandb, random backbone (no pretrained_ckpt needed for debug).
# Usage: sbatch sudoku_backplay_hard_debug.sh

export WANDB_API_KEY="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"

cd /home/frankwu2/PUMA
source /data/user_data/frankwu2/PUMA/venv/bin/activate
mkdir -p logs

echo "=== BackPlay fine-tuning debug run ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Start: $(date)"

/data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m backplay.train \
    --config-path ../yaml_files/sudoku_hard \
    --config-name backplay_finetune_debug \
    backplay.pretrained_ckpt=none \
    +training.ckpt_root=/data/user_data/frankwu2/PUMA/checkpoints \
    validation.val_dir=/data/user_data/frankwu2/PUMA/data/sudoku_hard \
    data.sudoku_type=preprocessed

TRAIN_EXIT=$?
echo "=== Done: $(date)  Exit: ${TRAIN_EXIT} ==="
exit ${TRAIN_EXIT}
