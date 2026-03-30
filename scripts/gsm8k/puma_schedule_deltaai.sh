#!/bin/bash
#SBATCH --job-name=gsm8k_puma_schedule
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x/job_%j.out
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh

#SBATCH --chdir=/u/zzhou24/projects/PUMA

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load default
module load cuda/12.9.0
source ~/miniconda3/bin/activate
conda activate smdm2

# Find a free port
find_free_port() {
  local port=$1
  while ss -tuln | grep -q ":$port "; do
    port=$((port + 1))
  done
  echo $port
}
PORT=$(find_free_port 29500)

export PYTHONUNBUFFERED=1
MASTER_ADDR=localhost

echo "PORT: $PORT"

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$PORT \
  -m maze.train_smdm_schedule --config-path "../yaml_files/gsm8k_smdm" --config-name train_schedule \
    model.pretrain_path=/projects/bgqz/zzhou24/models/mdm-170M-100e18.safetensors \
    data.gsm8k_test_path=/projects/bgqz/zzhou24/data/gsm8k/test.jsonl \
    data.data_dir=/projects/bgqz/zzhou24/data/gsm8k \
    +training.ckpt_root=/projects/bgqz/zzhou24/checkpoints \
    wandb.name="schedule-170M-100e18"
