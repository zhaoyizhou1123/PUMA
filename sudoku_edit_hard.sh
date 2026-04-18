#!/bin/bash
#SBATCH --job-name=puma_sudoku_edit_hard
#SBATCH --partition=general
#SBATCH --output=logs/train_sudoku_edit_hard_%j.log
#SBATCH --error=logs/train_sudoku_edit_hard_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=128GB
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-20

# ============================================================================
# PUMA Training Script - Sudoku Edit (progressive_edit, hard)
# ============================================================================

# ============================================================================
# MODIFY THESE PARAMETERS
# ============================================================================
export WANDB_API_KEY="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"

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

source /data/user_data/frankwu2//PUMA/venv/bin/activate

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
echo "Starting PUMA Training - Sudoku Edit (hard)"
echo "=========================================="
echo "Training starts at: $(date)"
echo ""

/data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m maze.train \
    --config-path ../yaml_files/sudoku_hard \
    --config-name base \
    training.strategy=progressive_edit \
    training.eval_steps=5000 \
    training.save_steps=5000 \
    +training.ckpt_root=/data/user_data/frankwu2/PUMA/checkpoints \
    validation.val_dir=/data/user_data/frankwu2/PUMA/data/sudoku_hard \
    validation.sampling.unmasking_num=[1] \
    +validation.sampling.edit_freq=[3] \
    +validation.sampling.edit_step=[4] \
    validation.track=False \
    data.seed=123 \
    data.sudoku_type=preprocessed \
    +training.max_steps=500000

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
