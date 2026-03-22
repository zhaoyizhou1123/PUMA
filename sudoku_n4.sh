#!/bin/bash
#SBATCH --job-name=puma_sudoku
#SBATCH --partition=general
#SBATCH --output=logs/train_sudoku_n4_%j.log
#SBATCH --error=logs/train_sudoku_n4_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=32GB

# ============================================================================
# PUMA Training Script - Sudoku
# ============================================================================

# ============================================================================
# MODIFY THESE PARAMETERS
# ============================================================================
CONFIG_FILE="yaml_files/sudoku_puma_n4.yaml"
export WANDB_API_KEY="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"
# Options: sudoku_puma.yaml, sudoku_baseline.yaml

# ============================================================================
# SLURM JOB INFORMATION
# ============================================================================
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
echo "=========================================="
echo "Setting Up Environment"
echo "=========================================="

cd /home/frankwu2/PUMA

source venv/bin/activate

mkdir -p logs

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# ============================================================================
# TRAINING
# ============================================================================
echo "=========================================="
echo "Starting PUMA Training - Sudoku"
echo "=========================================="
echo "Config File: ${CONFIG_FILE}"
echo "Training starts at: $(date)"
echo ""

# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 train.py --cfg ${CONFIG_FILE}

TRAIN_EXIT=$?

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: ${TRAIN_EXIT}"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "Training failed with exit code ${TRAIN_EXIT}"
    exit ${TRAIN_EXIT}
fi

echo "Training completed successfully!"
exit 0
