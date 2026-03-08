# !/bin/bash
export CUDA_VISIBLE_DEVICES=2

MASTER_ADDR=localhost
MASTER_PORT=29503

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train_proseco \
     training.strategy=proseco \
     +validation.sampling.edit_freq=[5,10] \
     +validation.sampling.edit_step=[1,2,4]
     