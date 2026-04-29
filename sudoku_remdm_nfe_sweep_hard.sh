#!/bin/bash
#SBATCH --job-name=puma_remdm_nfe_sweep
#SBATCH --partition=general
#SBATCH --output=logs/eval_remdm_nfe_sweep_%j.log
#SBATCH --error=logs/eval_remdm_nfe_sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64GB
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-m9-20

# ============================================================================
# ReMDM NFE sweep — num_steps matched to gibbs_edit NFEs
#
# gibbs_edit NFE = 81 * (edit_step + 1), edit_step in {0, 25, 50, ..., 200}
# → num_steps: 81, 2106, 4131, 6156, 8181, 10206, 12231, 14256, 16281
#
# Uses best-performing variant (rescale, eta=0.2) for tractable runtime (~12h).
# ============================================================================

export WANDB_API_KEY="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"

PRETRAINED_CKPT="/data/user_data/frankwu2/PUMA/checkpoints/sudoku_hard-pretraining-corrected/standard-hard-s123_date2026-03-30-22-47/step500000.pt"

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Start:  $(date)"
echo "Backbone ckpt: ${PRETRAINED_CKPT}"
echo ""

cd /home/frankwu2/PUMA
source /data/user_data/frankwu2/PUMA/venv/bin/activate
mkdir -p logs

echo "Python:  $(/data/user_data/frankwu2/PUMA/venv/bin/python --version)"
echo "PyTorch: $(/data/user_data/frankwu2/PUMA/venv/bin/python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(/data/user_data/frankwu2/PUMA/venv/bin/python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# gibbs_edit NFEs: 81*(k+1) for edit_step k in {0, 25, 50, 75, 100, 125, 150, 175, 200}
NUM_STEPS_LIST=(81 162 243 405 729 1377 2673 5265 10449 20817)

echo "=========================================="
echo "Starting ReMDM NFE Sweep (hard)"
echo "=========================================="

for NUM_STEPS in "${NUM_STEPS_LIST[@]}"; do
    echo ""
    echo "--- num_steps=${NUM_STEPS} ---"
    /data/user_data/frankwu2/PUMA/venv/bin/torchrun --nproc_per_node=1 -m remdm.eval \
        --cfg yaml_files/sudoku_hard/remdm_nfe_sweep.yaml \
        --ckpt "${PRETRAINED_CKPT}" \
        validation.sampling.num_steps=${NUM_STEPS}

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: num_steps=${NUM_STEPS} failed with exit code ${EXIT_CODE}"
        exit ${EXIT_CODE}
    fi
done

echo ""
echo "End: $(date)"
echo "All num_steps complete."
