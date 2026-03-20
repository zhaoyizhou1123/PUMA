# !/bin/bash
export CUDA_VISIBLE_DEVICES=1
MASTER_ADDR=localhost
MASTER_PORT=29501

for K in 8; do
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train --config-path "../yaml_files/sudoku_hard" --config-name base \
     training.strategy=edit_v3_3 \
     training.eval_steps=5000 \
     training.save_steps=5000 \
     training.K=$K \
     training.logging_steps=1 \
     validation.sampling.unmasking_num=[1] \
     +validation.sampling.edit_freq=[3] \
     +validation.sampling.edit_step=[4] \
     data.seed=123 \
     wandb.name=K${K}_edit_v3_3_s123
done