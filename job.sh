#!/bin/bash
#SBATCH --job-name=puma_tinygsm
#SBATCH --partition=general
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=360GB

# ============================================================================
# PUMA Training Script
# ============================================================================

# ============================================================================
# MODIFY THESE PARAMETERS
# ============================================================================
CONFIG_FILE="yaml_files/tinygsm_puma.yaml"
# Options: tinygsm_puma.yaml, tinygsm_baseline.yaml, sudoku_puma.yaml, sudoku_baseline.yaml
#          tinygsm_block_puma.yaml, tinygsm_block_baseline.yaml (use train_block.py for block)

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
# MULTI-NODE SETUP
# ============================================================================
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun -N1 -n1 -w "$MASTER_HOST" hostname -I | awk '{print $1}')
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

echo "=========================================="
echo "Multi-Node Configuration"
echo "=========================================="
echo "Master Host: ${MASTER_HOST}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "Number of Nodes: ${SLURM_NNODES}"
echo "GPUs per Node: 4"
echo ""

# ============================================================================
# TRAINING
# ============================================================================
echo "=========================================="
echo "Starting PUMA Training"
echo "=========================================="
echo "Config File: ${CONFIG_FILE}"
echo "Training starts at: $(date)"
echo ""

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --gpus-per-task=4 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train.py --cfg ${CONFIG_FILE}

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
