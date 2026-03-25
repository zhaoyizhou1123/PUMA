export CUDA_VISIBLE_DEVICES=4

MASTER_ADDR=localhost
MASTER_PORT=29509

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.eval --config-path "../yaml_files/sudoku_hard" --config-name base \
     training.strategy=progressive_edit \
     data.seed=123 \
     +validation.ckpt_path=ckpts/sudoku_hard-pretraining/K2_edit_v3_3_s123_date2026-03-15-15-47/step20000.pt \
     validation.sampling.unmasking_num=[1] \
     +validation.sampling.edit_freq=[3] \
     +validation.sampling.edit_step=[4] \
     validation.track=False

# torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --node_rank=0 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   -m maze.eval --config-path "../yaml_files/sudoku_hard" --config-name base \
#      training.strategy=progressive_edit \
#      data.seed=2028 \
#      +validation.ckpt_path=ckpts/sudoku_hard-pretraining/progressive_edit-s2028_date2026-03-10-06-30/step145000.pt \
#      validation.sampling.unmasking_num=[1] \
#      +validation.sampling.strategy=multi_proseco \
#      +validation.sampling.correction_step=0 \
#      validation.track=True

# torchrun \
# --nnodes=1 \
# --nproc_per_node=1 \
# --node_rank=0 \
# --rdzv_backend=c10d \
# --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
# -m maze.eval --config-path "../yaml_files/sudoku_hard" --config-name base \
#     training.strategy=progressive_edit \
#     data.seed=2028 \
#     +validation.ckpt_path=ckpts/sudoku_hard-pretraining/progressive_edit-s2028_date2026-03-10-06-30/step145000.pt \
#     validation.sampling.unmasking_num=[1] \
#     +validation.sampling.edit_freq=[1] \
#     +validation.sampling.edit_step=[0] \
#     validation.track=True