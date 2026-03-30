#!/bin/bash
#SBATCH --job-name=puma_sudoku_standard_hard
#SBATCH --partition=general
#SBATCH --output=logs/train_sudoku_standard_hard_%j.log
#SBATCH --error=logs/train_sudoku_standard_hard_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=128GB

# ============================================================================
# Standard MDM pre-training on sudoku_hard — PRISM Phase 1 baseline
#
# This produces a pretrained backbone checkpoint that PRISM fine-tuning
# (sudoku_prism_hard.sh) loads via prism.pretrained_ckpt.
# ============================================================================

export WANDB_API_KEY="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"

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Start:  $(date)"
echo ""

cd /home/frankwu2/PUMA
source venv/bin/activate
mkdir -p logs

echo "Python:  $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "=========================================="
echo "Starting Standard MDM Training (hard)"
echo "=========================================="

/data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m maze.train \
    --config-path ../yaml_files/sudoku_hard \
    --config-name base \
    training.strategy=standard \
    training.eval_steps=1000 \
    training.save_steps=5000 \
    +training.max_steps=500000 \
    +training.ckpt_root=/data/user_data/frankwu2/PUMA/checkpoints \
    validation.val_dir=/data/user_data/frankwu2/PUMA/data/sudoku_hard \
    validation.sampling.unmasking_num=[1] \
    validation.track=False \
    wandb.project=sudoku_hard-pretraining-corrected \
    "wandb.name=standard-hard-s\${data.seed}" \
    data.seed=123 \
    data.sudoku_type=preprocessed

TRAIN_EXIT=$?
echo "End: $(date)  Exit: ${TRAIN_EXIT}"
exit ${TRAIN_EXIT}
