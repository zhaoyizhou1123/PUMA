#!/bin/bash
#SBATCH --job-name=puma_sudoku_prism_recovery
#SBATCH --partition=general
#SBATCH --output=logs/train_sudoku_prism_recovery_%j.log
#SBATCH --error=logs/train_sudoku_prism_recovery_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=128GB
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-20

# ============================================================================
# PRISM recovery run — reverts Apr9→Apr18 changes to reproduce the Apr 9 result.
#
# Differences from prism/ (current):
#   algorithm.py: argmax-based token filling, matching the historical PRISM path
#   train.py:     recovery package now honors train/eval unmask modes from config
#   sampling.py:  preserved historical remasking behavior while allowing mode wiring
# ============================================================================

export WANDB_API_KEY="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"

PRETRAINED_CKPT="/data/user_data/frankwu2/PUMA/checkpoints/sudoku_hard-pretraining-corrected/standard-hard-s123_date2026-03-30-22-47/step500000.pt"

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Start:  $(date)"
echo "Backbone ckpt: ${PRETRAINED_CKPT}"
echo ""

cd /home/frankwu2/PUMA
source /data/user_data/frankwu2/PUMA/venv/bin/activate
mkdir -p logs

echo "Python:  $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "=========================================="
echo "Starting PRISM Recovery Fine-tuning (hard)"
echo "=========================================="

/data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m prism_recovery.train \
    --config-path ../yaml_files/sudoku_hard \
    --config-name prism_finetune \
    prism.pretrained_ckpt="${PRETRAINED_CKPT}" \
    prism.train_unmask_mode=top_k \
    prism.eval_unmask_mode=top_k \
    training.eval_steps=1000 \
    training.save_steps=5000 \
    training.max_steps=50000 \
    training.learning_rate=1e-4 \
    +training.ckpt_root=/data/user_data/frankwu2/PUMA/checkpoints \
    validation.val_dir=/data/user_data/frankwu2/PUMA/data/sudoku_hard \
    data.seed=123 \
    data.sudoku_type=preprocessed

TRAIN_EXIT=$?
echo "End: $(date)  Exit: ${TRAIN_EXIT}"
exit ${TRAIN_EXIT}
