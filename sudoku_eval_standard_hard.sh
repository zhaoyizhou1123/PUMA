#!/bin/bash
#SBATCH --job-name=puma_sudoku_eval_standard
#SBATCH --partition=general
#SBATCH --output=logs/eval_sudoku_standard_hard_%j.log
#SBATCH --error=logs/eval_sudoku_standard_hard_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64GB
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-20

CKPT="/data/user_data/frankwu2/PUMA/checkpoints/sudoku_hard-pretraining-corrected/standard-hard-s123_date2026-03-30-22-47/step500000.pt"

echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Start: $(date)"
echo "Ckpt: ${CKPT}"

cd /home/frankwu2/PUMA
source /data/user_data/frankwu2/PUMA/venv/bin/activate

/data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m maze.eval \
    --config-path ../yaml_files/sudoku_hard \
    --config-name base \
    +validation.ckpt_path="${CKPT}" \
    validation.track=false \
    validation.val_dir=/data/user_data/frankwu2/PUMA/data/sudoku_hard \
    hydra.run.dir=.

EXIT=$?
echo "End: $(date)  Exit: ${EXIT}"
exit ${EXIT}
