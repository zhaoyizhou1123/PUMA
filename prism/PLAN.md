# PRISM Baseline Implementation Plan

## Overview

We are re-implementing the PRISM algorithm (arXiv:2510.01384) inside the PUMA codebase to benchmark against our `progressive_edit` strategy on the `sudoku_hard` dataset.

**Key principle**: All new PRISM code lives in `prism/` and `yaml_files/sudoku_hard/prism_*.yaml`. Existing files are left untouched.

---

## Algorithm Summary

PRISM is a two-phase framework:

**Phase 1 — Standard MDM pretraining**
Train a standard masked diffusion model. This is a shared prerequisite: other baselines (e.g. pure MDM) can reuse the same pretrained backbone.

**Phase 2 — PRISM adapter fine-tuning**
Freeze the backbone. Train a lightweight adapter head that predicts per-token binary quality (is this token correct?).
- Sample x₀ from data; mask randomly → xₜ
- Backbone(xₜ) → logits + hidden states (frozen, no grad)
- One-step greedy unmask: reveal top-k confident masked tokens → x_s
- Binary labels: `(x_s == x₀)` at updated positions
- Self-correction loss: `BCE(adapter(hidden)[updated], labels)`
- Optional regularisation: `CE(logits[masked], x₀[masked])` × λ

**Inference (PRISM sampling)**
Standard iterative MDM decoding + adapter-guided remasking:
- Each step: unmask top-k confident masked tokens (same as mdm_sampling)
- Additionally: adapter scores every already-decoded token; remask the lowest-confidence ones so the model can correct them in subsequent steps

---

## File Structure

```
prism/
  __init__.py          # empty package marker
  model.py             # PrismMDMTransformer, AdapterHead
  algorithm.py         # prism_training_step()
  sampling.py          # prism_sampling(), prism_evaluate_ddp_sudoku()
  train.py             # training entry point (mirrors maze/train.py)
  PLAN.md              # this file

yaml_files/sudoku_hard/
  base.yaml            # [EXISTING] used as-is for standard pretraining
  prism_finetune.yaml  # [NEW] PRISM fine-tuning config (adds prism: section)

sudoku_standard_hard.sh   # [NEW] Phase 1: standard MDM pretraining
sudoku_prism_hard.sh      # [NEW] Phase 2: PRISM fine-tuning
```

No existing files are modified.

---

## Implementation Plan & Progress

### Step 1 — Standard MDM pretraining script  ✅ DONE
- `sudoku_standard_hard.sh`: runs `maze.train` with `training.strategy=standard`
- Produces a backbone checkpoint for Phase 2
- Same architecture and data as `sudoku_edit_hard.sh` for a fair comparison

### Step 2 — PrismMDMTransformer  ✅ DONE
- `prism/model.py`
- `AdapterHead`: 2-layer MLP (hidden_size → adapter_hidden → 1), GELU activation
- `PrismMDMTransformer(MDMTransformer)`:
  - New `forward_with_hidden(input_ids)` → `(logits, hidden_states)`
  - `adapter: AdapterHead` submodule
  - `forward()` unchanged — keeps full compatibility with `mdm_sampling`, `evaluate_ddp_sudoku`, etc.

### Step 3 — PRISM training algorithm  ✅ DONE
- `prism/algorithm.py` → `prism_training_step()`
- Pure function; no stateful buffer (unlike PhasedMasking)
- Handles backbone freezing, one-step unmask, BCE loss, optional reg loss

### Step 4 — PRISM sampling + eval  ✅ DONE
- `prism/sampling.py`
- `prism_sampling()`: drop-in replacement for `mdm_sampling` that also applies adapter-guided remasking
- `prism_evaluate_ddp_sudoku()`: mirrors `evaluate_ddp_sudoku` but uses `prism_sampling`

### Step 5 — prism/train.py  ✅ DONE
- Full training entry point: DDP setup, data loading, optimizer, scheduler, checkpointing, WandB logging
- Loads backbone from `prism.pretrained_ckpt`, freezes it, trains adapter only
- Evaluates both standard accuracy (mdm_sampling) and PRISM accuracy (prism_sampling) periodically

### Step 6 — YAML config  ✅ DONE
- `yaml_files/sudoku_hard/prism_finetune.yaml`: all required fields + new `prism:` section

### Step 7 — PRISM fine-tuning shell script  ✅ DONE
- `sudoku_prism_hard.sh`: loads pretrained ckpt, runs `prism.train`

---

## Next Steps (TODO)

- [x] **Smoke test**: all imports, shapes, and training step verified OK
- [x] **Debug Phase 1**: `sbatch sudoku_standard_hard_debug.sh` — 50 steps ✅
- [x] **Debug Phase 2**: `sbatch sudoku_prism_hard_debug.sh` — 50 steps, then fix OmegaConf struct errors ✅
- [x] **Fix sampling bug**: unmask `unmasking_num + num_remask` per step, then remask `num_remask` (net progress = `unmasking_num`)
- [x] **Align with paper hyperparams**: `num_demasking_tokens=4`, `reg_lambda=5.0`, `tune_backbone=true`, `batch_size=256`, `max_steps=20k`
- [ ] **Cancel bad PRISM run**: `scancel 6884048` — used old config, `prism_acc=0.0`
- [ ] **Submit Phase 1**: `sbatch sudoku_standard_hard.sh` — wait for backbone to converge
- [ ] **Submit Phase 2**: once Phase 1 plateaus, update `PRETRAINED_CKPT` in `sudoku_prism_hard.sh` and resubmit
- [ ] **Results comparison**: compare `prism_acc` vs `progressive_edit` solve accuracy on wandb

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Adapter architecture | 2-layer MLP on backbone hidden states | Simple, effective; matches PRISM `input_type: embedding` variant |
| Backbone time conditioning | None (PUMA has no time embeddings) | Adapter gets masking ratio implicitly through hidden state patterns |
| Streaming buffer | Not used for fine-tuning | PRISM fine-tuning is per-batch; no phased state needed |
| Eval during fine-tuning | Both standard + PRISM sampling | Track (a) backbone health and (b) adapter benefit separately |
| Separation from existing code | All new code in `prism/` | User requirement; zero modifications to existing files |
