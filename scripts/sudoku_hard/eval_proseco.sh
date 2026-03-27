export CUDA_VISIBLE_DEVICES=2

MASTER_ADDR=localhost
MASTER_PORT=29501

logfile=eval_proseco_gibbs.log

EDIT_START=0
for EDIT_STEP in 25 50 75 100 125 150 175 200; do
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
       +validation.ckpt_path=ckpts/sudoku_hard-pretraining-corrected/proseco-s123_date2026-03-15-17-48/step145000.pt \
       validation.sampling.unmasking_num=[1] \
       +validation.sampling.edit_freq=[1] \
       +validation.sampling.edit_step=[${EDIT_STEP}] \
       +validation.sampling.edit_strategy=gibbs_edit \
       validation.track=True >> $logfile 2>&1
done