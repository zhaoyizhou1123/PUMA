#!/bin/bash
#SBATCH --job-name=gsm8k-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
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

START_PORT=29500

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

echo "PORT: $PORT"

set -x

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$PORT \
  -m maze.eval_smdm --config-path "../yaml_files/gsm8k_smdm" --config-name smdm_eval \
    model.smdm_model_name=Diff_LLaMA_170M \
    data.test_ratio=1 \
    data.gsm8k_test_path=/projects/bgqz/zzhou24/data/gsm8k/test.jsonl \
    +data.gsm8k_cache_dir=/projects/bgqz/zzhou24/data/gsm8k \
    validation.ckpt_path="/projects/bgqz/zzhou24/checkpoints/gsm8k-smdm/1-progressive_edit-170M-100e18_date2026-03-29-15-25/step55000.pt" \
    +validation.track=True \
    validation.sampling.unmasking_num=[1,2,4,8] \
    +validation.sampling.edit_freq=[1] \
    +validation.sampling.edit_step=[4]