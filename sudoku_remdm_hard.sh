#!/bin/bash
#SBATCH --job-name=puma_sudoku_remdm_hard
#SBATCH --partition=general
#SBATCH --output=logs/eval_sudoku_remdm_hard_%j.log
#SBATCH --error=logs/eval_sudoku_remdm_hard_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64GB
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-20

# ============================================================================
# ReMDM evaluation on sudoku_hard — inference-only, no training.
#
# Reference: arXiv:2503.00307v4
#   "Remasking Discrete Diffusion Models"  (Kuleshov group)
#
# Prerequisite: a pretrained standard MDM backbone from sudoku_standard_hard.sh.
# Set PRETRAINED_CKPT below.
# ============================================================================

export WANDB_API_KEY="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"

# ============================================================
# SET THIS to the backbone checkpoint from Phase 1 pretraining
# ============================================================
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

echo "Python:  $(/data/user_data/frankwu2/PUMA/venv/bin/python --version)"
echo "PyTorch: $(/data/user_data/frankwu2/PUMA/venv/bin/python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(/data/user_data/frankwu2/PUMA/venv/bin/python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "=========================================="
echo "Starting ReMDM Evaluation (hard)"
echo "=========================================="

/data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m remdm.eval \
    --cfg yaml_files/sudoku_hard/remdm_eval.yaml \
    --ckpt "${PRETRAINED_CKPT}"

EVAL_EXIT=$?
echo "End: $(date)  Exit: ${EVAL_EXIT}"
exit ${EVAL_EXIT}
