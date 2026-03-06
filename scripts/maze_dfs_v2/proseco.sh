# !/bin/bash
export CUDA_VISIBLE_DEVICES=0

MASTER_ADDR=localhost
MASTER_PORT=29504

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train_proseco --config-name maze_dfs_v2 \
     training.strategy=proseco \
     training.eval_steps=5000 \
     training.save_steps=5000 \
     validation.sampling.unmasking_num=[10] \
     +validation.sampling.edit_freq=[1] \
     +validation.sampling.edit_step=[1] \
     validation.track=False
     