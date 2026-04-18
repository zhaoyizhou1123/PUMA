# ReMDM

**Checkpoint**: `standard-hard-s123_date2026-03-30-22-47/step500000.pt`  
**Reference**: arXiv:2503.00307 (Kuleshov group)

## Results

| Sampler | Accuracy |
|---|---|
| Standard (mdm_sampling, greedy top-k) | 0.528 |
| ReMDM cap η=0.0, freeze_unmasked=False | 0.101 |
| ReMDM cap η=0.2, freeze_unmasked=False | 0.165 |
| ReMDM cap η=0.0, freeze_unmasked=True  | ~0.10 |
| ReMDM conf, freeze_unmasked=False      | 0.156 |

## Analysis

The poor performance is not specific to ReMDM — it reflects a fundamental mismatch between **MDLM-style sampling** (which ReMDM builds on) and constraint satisfaction tasks like Sudoku.

**MDLM-style sampling** decodes masked tokens with a probability derived from the noise schedule, uniform across all positions regardless of confidence. Token values are sampled stochastically from `p_x0`. This is principled — it produces valid samples from the model's learned distribution — and works well for open-ended text generation where diversity matters.

**MDM-style (confidence-guided) sampling** always decodes the most confident masked token first, committing greedily via argmax. This is a heuristic with no probabilistic grounding, but it is a natural fit for CSPs: decode the most constrained cell first, which narrows uncertainty for subsequent cells.

For Sudoku, decoding order and determinism are critical. A wrong token placed early (due to uniform random ordering + stochastic sampling) corrupts the model's context for all subsequent steps. The standard sampler avoids this entirely. Training-sampling alignment (standard MDM training vs. MDLM schedule) is a secondary factor since the model has no time conditioning.

# Remedi

**Reference**: arXiv:2509.23653  
**Checkpoints**: `remedi-finetune-s123_date2026-04-01-12-06` (run 1, tune_backbone=True), and several ablation runs  
**Backbone**: `standard-hard-s123_date2026-03-30-22-47/step500000.pt` (std_acc = 0.528)

## Results

| Config | std_acc (end) | remedi_acc (end) | Notes |
|---|---|---|---|
| tune_backbone=True, incorrect_ratio=0.1 | 0.522 | 0.516 | baseline run |
| tune_backbone=False | 0.494 | 0.458 | worse, not better |
| backprop_warmup_steps=2000 | ~0.52 | ~0.50 | no improvement |
| incorrect_ratio=0.0 | higher early | no improvement | confirms distribution shift cause |

remedi_acc never meaningfully exceeds std_acc across all runs. Both converge below the pretrained baseline of 0.528.

## Ablation: isolating the std_acc crash

A sequence of controlled experiments identified the cause of the early accuracy drop (0.528 → ~0.46 at step 1000):

| What was blocked during warmup | std_acc at step 1000 |
|---|---|
| Nothing (original) | 0.460 |
| UPS write-back (backprop_linears frozen) | 0.420 |
| Write-back + L_ups→TPS gradient (detach) | 0.422 |
| Everything — skip L_diffusion too (TPS fully frozen) | ~0.528 |

**Conclusion**: the crash is caused entirely by L_diffusion fine-tuning TPS on the SFT distribution (which contains incorrect tokens). Write-back and L_ups gradients are not contributing factors.

## Analysis

Three compounding reasons RemeDi does not improve over the baseline.

**1. TPS is degraded by incorrect-token training.**
The SFT distribution injects incorrect tokens into the input, pulling TPS weights away from the masked-only optimum it was pretrained on. std_acc drops from 0.528 → ~0.46 immediately and only partially recovers by step 50k. The system is trying to improve on a baseline it has already damaged.

**2. The remasking signal is identical to standard MDM.**
With `use_ups=False` (matching the official inference.py), remasking ranks tokens by `logits[argmax_token]` — the same signal standard MDM already uses for selection. The only extra capability is evicting previously committed tokens, but that only helps if wrong tokens score lower as context grows. There is no explicit training pressure for TPS to become less confident at wrong positions over time.

**3. The UPS BCE does not learn a useful independent signal.**
For masked positions (the majority of training signal), the soft label is `p_θ(x₀|x_t)` — TPS's own confidence on the same input in the same forward pass. UPS just learns to replicate TPS, adding nothing new. For incorrect tokens (`y=0`, the genuinely useful case), these are rare (~10% peak due to `ρ_t,incorrect = 4·0.1·t·(1-t)`), so their gradient is swamped by the soft-target majority. Net result: UPS confidence ≈ TPS logit at argmax, so `use_ups=True` ≈ `use_ups=False` ≈ standard MDM ranking.

**Underlying structural limitation.**
RemeDi was designed for large LLMs on language tasks where (a) the pretrained model has substantial headroom, (b) errors are locally detectable so UPS can learn a signal beyond TPS, and (c) the model has enough capacity for UPS to learn independently. A 10.5M parameter model near its capacity ceiling on a globally-constrained combinatorial task hits all three failure modes simultaneously.

# PRISM

**Reference**: arXiv:2510.01384  
**Checkpoints**: `prism-finetune-s123_date2026-04-01-12-06` (run 6923992), `prism-finetune-s123_date2026-04-08-14-13` (run 7017836)  
**Backbone**: `standard-hard-s123_date2026-03-30-22-47/step500000.pt` (std_acc = 0.528)

## Results

| Config | train_unmask | eval_unmask | std_acc (end) | prism_acc (end) | Notes |
|---|---|---|---|---|---|
| old code (6923992) | top_k | top_k | ~0.499 | ~0.499 | No lift over std_acc |
| new code (7017836) | random | random | ~0.499 | ~0.360 | Adapater undertrained |
| new code + eval=top_k | random | top_k | ~0.499 | ~0.499 | No lift over std_acc |

`prism_acc` never meaningfully exceeds `std_acc` across all unmasking strategy combinations.

## Analysis

Three compounding hypotheses for why PRISM's adapter provides no lift.

**1. BCE is a weak self-correction signal.**

The adapter is trained with:

$$\mathcal{L}_{SC} = \text{BCE}(\hat{g}_\phi(y^i), \;\mathbf{1}[x^i = y^i])$$

The binary label (correct or not) is a very indirect proxy for what the adapter needs to do at inference: **rank** all clean tokens by how likely they are to be wrong, so the lowest-ranked k can be remasked. BCE trains the adapter to predict an absolute probability at isolated positions with no contrastive signal — it never learns relative ordering between positions. A token with 60% confidence and one with 95% confidence produce similar BCE gradients in the same direction, even though they should receive opposite remasking priorities.

Furthermore, with a strong backbone, the vast majority of filled tokens are correct, so labels are heavily skewed toward 1. BCE loss has a trivial near-zero solution: predict ~1.0 everywhere. The adapter collapses.

**2. Backbone and adapter operate in disjoint regimes.**

During training, `x_s` is produced by one forward pass from `x_t` — the backbone fills in k tokens from a partially masked sequence. At inference, the adapter scores tokens in a sequence built up over many sequential decoding steps with very different distributional properties. The adapter's training signal (sparse, noisy `x_s` from a single step) does not match what it sees at inference (densely decoded sequence from many steps).

**3. PRISM's BCE vs. progressive_edit's CE: structural difference.**

`progressive_edit` trains the backbone directly with CE loss on sequences containing its own past prediction errors — the model learns to predict correct tokens from partially-filled contexts. This is the same objective the backbone was pretrained with, requiring no additional adapter.

PRISM decouples correction into two sequential steps: (a) a frozen backbone predicts tokens, (b) an adapter learns to detect errors via BCE. BCE can only signal "this token might be wrong" — it cannot tell the backbone what the right token should be. After remasking, the backbone simply re-predicts from scratch with no signal that the previous answer was wrong beyond the adapter's noisy 0/1 label.

`progressive_edit` sidesteps this bottleneck entirely: the backbone is trained to predict correctly *given* that errors exist in context, so the correction signal is implicit in CE rather than mediated through a sparse binary proxy.

---

# Backplay

**Reference**: arXiv:2601.06428  
**Checkpoints**: `backplay-finetune-s123_date2026-04-07-20-25` (run 1), `...-04-08-00-28` (run 2)  
**Backbone**: `standard-hard-s123_date2026-03-30-22-47/step500000.pt` (std_acc = 0.528)

## Results

| Config | backplay_acc (end) | Notes |
|---|---|---|
| LR=3e-4, error_weight=1.0 (paper BCE) | 0.519 | Converges smoothly, never exceeds baseline |
| LR=1e-3, error_weight=10 | 0.502 | Wildly fluctuating, worse than run 1 |

## Analysis

BackPlay never exceeds the standard sampler baseline (0.528) despite 50k fine-tuning steps. Two compounding issues explain this.

**1. Adapter calibration collapse.** The adapter is trained with BCE loss to predict P(token is correct). Among non-masked positions in z_t, ~99% of tokens are correct (the backbone solves 52.8% of puzzles, but failed puzzles typically have only 2–5 wrong tokens out of 81, so token-level accuracy ≈ 99%). BCE loss overwhelmingly rewards predicting "all correct," collapsing `avg_max_err` from ~0.64 → 0.03 over training. With τ=0.75, nothing ever exceeds the threshold and corrections effectively stop.

**2. Corrections are net negative even when they fire.** When the adapter does remask a token, it is not reliably identifying the ~1% of wrong tokens — it selects near-randomly among clean tokens. ~99% of clean tokens are correct, so a random remask breaks a correct token and re-samples it from a backbone that is far from perfect. The corrected 17% of sequences end up with lower solve rate than the uncorrected 83%, pulling overall accuracy below baseline.

Increasing `error_weight` slows the collapse but does not stop it, and a higher LR causes oscillation in `avg_remasked` that translates directly to oscillation in `backplay_acc`.

**3. Structural limitation: BCE vs. CE.** The deeper issue is that BackPlay splits correction into two sequential problems: a BCE adapter *detects* errors, then the backbone *fixes* them after remasking. Our `progressive_edit` strategy collapses both into one: the backbone is trained with plain CE loss on sequences that already contain its own past prediction errors (via `PhasedMaskingEdit`). This is a much more natural objective — the backbone was pre-trained with CE and already knows how to predict correct tokens from partially-filled contexts. BackPlay's adapter must learn error detection from scratch on a sparse signal, with a train/inference distribution mismatch (z_t ≠ xt_new), and requires careful calibration of τ. Progressive edit avoids all of these by keeping correction in the CE domain.
