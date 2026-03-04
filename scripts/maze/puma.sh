# !/bin/bash
export CUDA_VISIBLE_DEVICES=2

MASTER_ADDR=localhost
MASTER_PORT=29501

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train \
     training.strategy=progressive

