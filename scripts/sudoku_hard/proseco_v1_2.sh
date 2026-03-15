# !/bin/bash
export CUDA_VISIBLE_DEVICES=2

MASTER_ADDR=localhost
MASTER_PORT=29502

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train_proseco_v1_2 --config-path "../yaml_files/sudoku_hard" --config-name base \
     training.strategy=proseco \
     training.eval_steps=5000 \
     training.save_steps=5000 \
     training.logging_steps=1 \
     validation.sampling.unmasking_num=[1] \
     +validation.sampling.edit_freq=[3] \
     +validation.sampling.edit_step=[4] \
     validation.track=False \
     data.seed=123 \
     wandb.name=proseco_v1_2_s123
     