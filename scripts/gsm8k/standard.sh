#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

MASTER_ADDR=localhost
MASTER_PORT=29500
while ss -tlnp | grep -q ":${MASTER_PORT} "; do
  MASTER_PORT=$((MASTER_PORT + 1))
done
echo "Using MASTER_PORT=${MASTER_PORT}"

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train_smdm --config-path "../yaml_files/gsm8k_smdm" --config-name train_standard
