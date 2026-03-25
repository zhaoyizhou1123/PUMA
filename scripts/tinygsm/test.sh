#!/bin/bash
#SBATCH --job-name=tinygsm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%x/job_%j.out
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh

#SBATCH --chdir=/u/zzhou24/projects/PUMA

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load default
module load cuda/12.9.0
source ~/miniconda3/bin/activate
conda activate puma

# `module default` sets NCCL_NET_PLUGIN and adds the OFI plugin dir to
# LD_LIBRARY_PATH. The plugin has a bug: CXI domain creation fails (RC -38),
# and the cleanup triggers a C-heap double-free in all ranks.
# Unsetting NCCL_NET_PLUGIN alone isn't enough — NCCL also searches
# LD_LIBRARY_PATH for libnccl-net.so. Strip it from both.
unset NCCL_NET_PLUGIN
LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/sw/user/nccl/' | tr '\n' ':' | sed 's/:*$//')
export LD_LIBRARY_PATH

START_PORT=29510

MASTER_ADDR=localhost

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$START_PORT \
  train.py --cfg yaml_files/tinygsm/base.yaml
