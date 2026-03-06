# !/bin/bash
export CUDA_VISIBLE_DEVICES=2

MASTER_ADDR=localhost
MASTER_PORT=29503

# torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --node_rank=0 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   -m maze.eval \
#      training.strategy=proseco \
#      +validation.ckpt_path=ckpts/maze-proseco-s123/step40000.pt \
#      validation.sampling.unmasking_num=[1,5] \
#      +validation.sampling.edit_freq=[1] \
#      +validation.sampling.edit_step=[1] \
#      validation.track=False

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m maze.eval \
     training.strategy=progressive \
     +validation.ckpt_path=ckpts/maze-puma-s123/step40000.pt \
     validation.sampling.unmasking_num=[10] \

# torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --node_rank=0 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   -m maze.eval \
#      training.strategy=progressive_edit \
#      +validation.ckpt_path=ckpts/maze-progressive_edit-s123_v0/step40000.pt \
#      validation.sampling.unmasking_num=[10] \
#      +validation.sampling.edit_freq=[1] \
#      +validation.sampling.edit_step=[1] \
#      validation.track=True