#!/bin/bash
#SBATCH --job-name=tinygsm
#SBATCH --nodes=2
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

unset NCCL_NET_PLUGIN
LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/sw/user/nccl/' | tr '\n' ':' | sed 's/:*$//')
export LD_LIBRARY_PATH

START_PORT=29510

# For multi-node: get the hostname of the first allocated node as master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# srun distributes one task per node; SLURM_NODEID is 0 on node0, 1 on node1
# total world_size = 2 nodes x 4 GPUs = 8; batch_size in config is the global total,
# so per-GPU batch = batch_size / 8 (vs batch_size / 4 with single node).
# To keep the same per-GPU batch size, double training.batch_size below.
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$START_PORT \
  -m maze.train --config-path "../yaml_files/tinygsm" --config-name base \
     training.strategy=progressive \
     training.batch_size=256 \
     training.eval_steps=5000 \
     training.save_steps=10000 \
     validation.sampling.unmasking_num=[2,3] \
     +training.ckpt_root=/projects/bgqz/zzhou24/checkpoints/ \
     +training.resume=/projects/bgqz/zzhou24/checkpoints/mdm-pretraining/puma_date2026-03-22-14-34/step50000.pt \
     data.seed=2026
