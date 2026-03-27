export CUDA_VISIBLE_DEVICES=3

MASTER_ADDR=localhost
MASTER_PORT=29511

# torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --node_rank=0 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   -m maze.eval --config-path "../yaml_files/sudoku_hard" --config-name base \
#      training.strategy=progressive_edit \
#      data.seed=123 \
#      +validation.ckpt_path=ckpts/sudoku_hard-pretraining-corrected/progressive_edit-s123_date2026-03-15-17-46/step145000.pt \
#      validation.sampling.unmasking_num=[1] \
#      +validation.sampling.edit_freq=[1] \
#      +validation.sampling.edit_step=[20] \
#      validation.track=False
EDIT_STEP=25
for EDIT_START in 1 2 0; do
  echo "=== edit_step=${EDIT_STEP}, edit_start=${EDIT_START} ===" >> eval_skip_start.log 2>&1
  torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m maze.eval --config-path "../yaml_files/sudoku_hard" --config-name base \
       training.strategy=progressive_edit \
       data.seed=123 \
       +validation.ckpt_path=ckpts/sudoku_hard-pretraining-corrected/progressive_edit-s123_date2026-03-15-17-46/step145000.pt \
       validation.sampling.unmasking_num=[1] \
       +validation.sampling.edit_freq=[1] \
       "+validation.sampling.edit_step=[${EDIT_STEP}]" \
       +validation.sampling.edit_strategy=gibbs_edit \
       +validation.sampling.edit_start=${EDIT_START} \
       validation.track=True >> eval_skip_start.log 2>&1
done

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