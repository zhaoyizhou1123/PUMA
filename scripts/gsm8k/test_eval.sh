#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

MASTER_ADDR=localhost
MASTER_PORT=29501

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.eval_smdm --config-path "../yaml_files/gsm8k_smdm" --config-name smdm_eval \
    model.smdm_model_name=Diff_LLaMA_170M \
    data.test_ratio=0.1 \
    validation.ckpt_path="/home/zhaoyiz/projects/PUMA/ckpts/gsm8k-smdm/standard-170M_date2026-03-28-01-29/step50000.pt"
