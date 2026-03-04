#!/bin/bash
#SBATCH --job-name=gen_sudoku_n4
#SBATCH --partition=cpu
#SBATCH --output=logs/gen_sudoku_n4_%j.log
#SBATCH --error=logs/gen_sudoku_n4_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --qos=cpu_qos

# ============================================================================
# Sudoku n=4 Data Generation
# ============================================================================

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
cd /home/frankwu2/PUMA

source venv/bin/activate

mkdir -p logs

echo "Python version: $(python --version)"
echo ""

# ============================================================================
# DATA GENERATION
# ============================================================================
echo "=========================================="
echo "Starting Sudoku n=4 Generation"
echo "=========================================="
echo "Generation starts at: $(date)"
echo ""

python data/generate_sudoku.py \
    --n 4 \
    --num-train 1000000 \
    --num-test 100000   \
    --workers 128 \
    --difficulty-min 0.6 \
    --difficulty-max 0.78 \
    --allow-multi-solutions \
    --timeout 1

GEN_EXIT=$?

echo ""
echo "=========================================="
echo "Generation Complete"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: ${GEN_EXIT}"

if [ ${GEN_EXIT} -ne 0 ]; then
    echo "Generation failed with exit code ${GEN_EXIT}"
    exit ${GEN_EXIT}
fi

echo "Generation completed successfully!"
exit 0
