#!/bin/bash
#SBATCH --job-name=gsm8k_edit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/%x/job_%j_%a.out
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh
#SBATCH --array=0

#SBATCH --chdir=/u/zzhou24/projects/PUMA

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load default
module load cuda/12.9.0
source ~/miniconda3/bin/activate
conda activate smdm2

# Define your parameters
# SEEDS=(123)
START_PORT=29500

# SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Find a free port starting from START_PORT
find_free_port() {
  local port=$1
  while ss -tuln | grep -q ":$port "; do
    port=$((port + 1))
  done
  echo $port
}
PORT=$(find_free_port $((START_PORT + SLURM_ARRAY_TASK_ID)))

export PYTHONUNBUFFERED=1
MASTER_ADDR=localhost

echo "Running SEED: $SEED | PORT: $PORT"

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$PORT \
  -m maze.train_smdm --config-path "../yaml_files/gsm8k_smdm" --config-name train_schedule \
    training.strategy=progressive_edit \
    +validation.sampling.edit_freq=[1] \
    +validation.sampling.edit_step=[1] \
    training.batch_size=256 \
    training.batch_size_per_gpu=256 \
    training.num_epochs=20 \
    training.K=2 \
    model.pretrain_path=/projects/bgqz/zzhou24/models/mdm-170M-10e18.safetensors \
    data.gsm8k_test_path=/projects/bgqz/zzhou24/data/gsm8k/test.jsonl \
    data.data_dir=/projects/bgqz/zzhou24/data/gsm8k \
    +training.ckpt_root=/projects/bgqz/zzhou24/checkpoints
