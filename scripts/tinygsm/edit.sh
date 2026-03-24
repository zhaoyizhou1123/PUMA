#!/bin/bash
#SBATCH --job-name=tinygsm_edit
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
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

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

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$START_PORT \
  -m maze.train --config-path "../yaml_files/tinygsm" --config-name base \
     training.strategy=progressive_edit \
     training.batch_size=256 \
     training.eval_steps=1 \
     training.save_steps=10000 \
     validation.sampling.unmasking_num=[2,3] \
     +validation.sampling.edit_freq=[3] \
     +validation.sampling.edit_step=[4] \
     +training.ckpt_root=/projects/bgqz/zzhou24/checkpoints/ \
     wandb.name=puma-edit \
     data.seed=2026
