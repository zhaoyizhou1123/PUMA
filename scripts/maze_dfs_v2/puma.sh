# !/bin/bash
export CUDA_VISIBLE_DEVICES=0

MASTER_ADDR=localhost
MASTER_PORT=29502

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train --config-name maze_dfs_v2 \
     training.strategy=progressive \
     training.eval_steps=5000 \
     training.save_steps=5000 \
     validation.sampling.unmasking_num=[10] \
     validation.track=False
