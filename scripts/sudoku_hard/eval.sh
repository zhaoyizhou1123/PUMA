export CUDA_VISIBLE_DEVICES=9

MASTER_ADDR=localhost
MASTER_PORT=29509

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.eval --config-path "../yaml_files/sudoku_hard" --config-name base \
     training.strategy=proseco \
     +validation.ckpt_path=ckpts/sudoku_hard-pretraining/proseco-s2027_date2026-03-10-05-42/step145000.pt \
     validation.sampling.unmasking_num=[1] \
     +validation.sampling.edit_freq=[3] \
     +validation.sampling.edit_step=[4] \
     validation.track=True

