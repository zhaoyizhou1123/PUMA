export CUDA_VISIBLE_DEVICES=3

MASTER_ADDR=localhost
MASTER_PORT=29510

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
logfile=eval_gibbs_random_step50k.log

EDIT_START=0
for EDIT_STEP in 0 25 50 75 100 125 150 175 200; do
  echo "=== edit_step=${EDIT_STEP}, edit_start=${EDIT_START} ===" >> $logfile 2>&1
  torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m maze.eval --config-path "../yaml_files/sudoku_hard" --config-name base \
       training.strategy=progressive_edit \
       data.seed=123 \
       +validation.ckpt_path=ckpts/sudoku_hard-pretraining-corrected/progressive_edit-s123_date2026-03-15-17-46/step50000.pt \
       validation.sampling.unmasking_num=[1] \
       +validation.sampling.edit_freq=[1] \
       "+validation.sampling.edit_step=[${EDIT_STEP}]" \
       +validation.sampling.edit_strategy=gibbs_edit \
       +validation.sampling.edit_start=${EDIT_START} \
       validation.track=True >> $logfile 2>&1
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