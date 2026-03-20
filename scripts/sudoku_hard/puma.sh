# !/bin/bash
export CUDA_VISIBLE_DEVICES=4

MASTER_ADDR=localhost
MASTER_PORT=29500

for s in 123 2026 2027 2028; do
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train --config-path "../yaml_files/sudoku_hard" --config-name base \
     training.strategy=progressive \
     training.eval_steps=5000 \
     training.save_steps=5000 \
     validation.sampling.unmasking_num=[1] \
     validation.track=False \
     data.seed=$s
done