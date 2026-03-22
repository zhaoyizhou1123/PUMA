# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PUMA (Progressive Unmasking for Accelerated Masked Diffusion Training)** is a research implementation for training Masked Diffusion Models (MDMs) more efficiently. The core idea is to progressively unmask tokens during training using model confidence, aligning training and inference-time masking patterns to accelerate training. Paper: arXiv:2602.10314.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate puma
```

PyTorch is **not** in `environment.yml` — install separately for your CUDA version.

## Data Preparation

**Sudoku**: Download `sudoku-train-data.npy` and `sudoku-test-data.npy` from the Google Drive link in README.md and place in `data/sudoku_new/`.

**TinyGSM**:
```bash
python data/tiny_gsm.py  # generates labels.bin, meta.json, prompt_mask.bin
```

**Maze**:
```bash
python data/maze.py
```

## Running Training

**Single GPU (local)**:
```bash
python train.py --cfg yaml_files/sudoku_puma.yaml
python train.py --cfg yaml_files/tinygsm_puma.yaml
```

**Block diffusion variant** (slower, no KV-cache):
```bash
python train_block.py --cfg yaml_files/tinygsm_block_puma.yaml
```

**Multi-GPU via SLURM** (edit `account` and `partition` in `job.sh` first):
```bash
sbatch job.sh
```

**ARM initialization workflow**: First pretrain with `model.causal=true`, then set `model.arm_init=<checkpoint_path>` and `model.causal=false`.

Before running: update `wandb.entity` in the YAML config.

## Evaluation

```bash
python evaluate.py --cfg yaml_files/sudoku_puma.yaml
```

Evaluation also runs automatically during training at `eval_steps` intervals.

## Architecture

### Core Algorithm: `progressive.py`

The `PhasedMasking` class implements PUMA's streaming batch approach:
- Divides training into **K phases** with masking ratio intervals `[i/K, (i+1)/K]`
- Each step: sample target unmasking ratio from the current phase, run forward pass, select top-k confident tokens to unmask, advance sequence to next phase
- When a sequence completes all K phases, it is refilled from the dataset
- Loss functions: `mdm_loss_fn` (masked positions only), `mdm_edit_loss_fn` (all non-prompt positions), `mdm_edit_loss_fn_v5` (hybrid)

### Model: `model/transformer.py`

`MDMTransformer` is a bidirectional Qwen2-style Transformer (no time embeddings, no causal mask):
- `MDMConfig` dataclass for all hyperparameters
- `QwenAttention`: grouped-query attention with RoPE positional embeddings
- `SwiGLU` FFN, `RMSNorm` pre-norm
- Optional tied embedding/LM head weights, optional causal mask (for ARM pretraining)

### Training entry points

| File | Purpose |
|------|---------|
| `train.py` | Main MDM training — Sudoku and TinyGSM |
| `train_block.py` | Block diffusion training |
| `train_edit.py` | Edit/instruction-following training |
| `train_proseco.py` | Proseco baseline |

### Sampling: `sampling.py`

Two strategies:
- `mdm_sampling`: iterative bidirectional denoising (confidence-based token selection)
- `arm_sampling`: autoregressive left-to-right generation

### Configuration: `yaml_files/`

YAML files use OmegaConf. Key sections:
- `model`: architecture hyperparameters
- `training`: `strategy` (`"progressive"` for PUMA, `"baseline"` for vanilla MDM, `"arm"`), `K` (number of phases), `confidence_threshold`, learning rate, batch size
- `data`: dataset type, paths, `mask_id`
- `validation`: sampling configs evaluated during training
- `wandb`: logging settings

`k_schedule` in training config allows dynamically changing K during training as a list of `[K_value, step]` pairs.

### Checkpoints

Saved to `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/` as `step{N}.pt` and `step{N}_ema.pt`.