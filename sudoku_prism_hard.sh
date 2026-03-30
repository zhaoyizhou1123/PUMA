#!/bin/bash
#SBATCH --job-name=puma_sudoku_prism_hard
#SBATCH --partition=general
#SBATCH --output=logs/train_sudoku_prism_hard_%j.log
#SBATCH --error=logs/train_sudoku_prism_hard_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=128GB

# ============================================================================
# PRISM fine-tuning on sudoku_hard — Phase 2
#
# Prerequisite: run sudoku_standard_hard.sh first to get a pretrained backbone.
# Set PRETRAINED_CKPT below to the desired checkpoint from that run.
# ============================================================================

export WANDB_API_KEY="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"

# ============================================================
# SET THIS to the backbone checkpoint from Phase 1 pretraining
# ============================================================
PRETRAINED_CKPT="/data/user_data/frankwu2/PUMA/checkpoints/sudoku_hard-pretraining-corrected/standard-hard-s123_date2026-03-29-19-46/step145000.pt"

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Start:  $(date)"
echo "Backbone ckpt: ${PRETRAINED_CKPT}"
echo ""

cd /home/frankwu2/PUMA
source venv/bin/activate
mkdir -p logs

echo "Python:  $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "=========================================="
echo "Starting PRISM Fine-tuning (hard)"
echo "=========================================="

/data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m prism.train \
    --config-path ../yaml_files/sudoku_hard \
    --config-name prism_finetune \
    prism.pretrained_ckpt="${PRETRAINED_CKPT}" \
    training.eval_steps=1000 \
    training.save_steps=5000 \
    +training.ckpt_root=/data/user_data/frankwu2/PUMA/checkpoints \
    validation.val_dir=/data/user_data/frankwu2/PUMA/data/sudoku_hard \
    data.seed=123 \
    data.sudoku_type=preprocessed

TRAIN_EXIT=$?
echo "End: $(date)  Exit: ${TRAIN_EXIT}"
exit ${TRAIN_EXIT}
