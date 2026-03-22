# !/bin/bash
export CUDA_VISIBLE_DEVICES=5

MASTER_ADDR=localhost
MASTER_PORT=29505

for seed in 1 12 123 1234; do
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.train --config-name maze_dfs \
     model.num_layers=4 \
     model.num_attention_heads=4 \
     model.num_kv_heads=2 \
     training.strategy=progressive_edit \
     training.eval_steps=5000 \
     training.save_steps=5000 \
     training.num_epochs=15 \
     validation.sampling.unmasking_num=[10] \
     +validation.sampling.edit_freq=[1] \
     +validation.sampling.edit_step=[1] \
     validation.track=False \
     wandb.name="3M-maze-progressive_edit-s${seed}" \
     wandb.project="3M-maze17x17-pretraining" \
     data.seed=$seed
done