# !/bin/bash
export CUDA_VISIBLE_DEVICES=2

MASTER_ADDR=localhost
MASTER_PORT=29501

for lr in 1e-4 5e-5 1e-5; do
  torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m maze.train --config-name maze_dfs_v2 \
       training.strategy=progressive_edit \
       training.eval_steps=5000 \
       training.save_steps=5000 \
       training.learning_rate=$lr \
       validation.sampling.unmasking_num=[10] \
       +validation.sampling.edit_freq=[1] \
       +validation.sampling.edit_step=[1] \
       validation.track=False \
       wandb.name="maze-progressive_edit-lr${lr}-s123"
done
     