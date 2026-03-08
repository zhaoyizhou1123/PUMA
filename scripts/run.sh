# !/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# conda init
# conda deactivate
# conda activate puma

# MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# MASTER_ADDR=$(srun -N1 -n1 -w "$MASTER_HOST" hostname -I | awk '{print $1}')
# MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
MASTER_ADDR=localhost
MASTER_PORT=29500

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_edit.py --cfg yaml_files/sudoku_puma_edit.yaml

# torchrun \
#   --nnodes=1 \
#   --nproc_per_node=2 \
#   --node_rank=0 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   train.py --cfg yaml_files/sudoku_puma.yaml

