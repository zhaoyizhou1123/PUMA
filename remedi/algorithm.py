"""
RemeDi SFT training step
=========================
Reference: "Don't Settle Too Early: Self-Reflective Remasking for Diffusion
Language Models" (arXiv:2509.23653), supervised fine-tuning — Algorithm 1.

Noise schedule
--------------
Per sequence, sample t ~ U[0,1):
  ρ_t,mask      = t
  ρ_t,incorrect = 4·r·t·(1−t),  r = 0.1

Input x_t (Algorithm 1, lines 4–5), non-prompt positions only:
  • each position masked with prob ρ_t,mask
  • among positions still unmasked, each replaced by a random wrong token
    with prob ρ_t,incorrect (Bernoulli, independent)

Prompt positions are never corrupted.

Inequality (3) in the paper holds for r ≤ 0.25, ensuring remasking incorrect
tokens does not increase total mask count in expectation vs. the schedule.

Losses (Algorithm 1)
--------------------
  ℒ_diffusion — Algorithm 1 line 8: per-seq (L/|𝒮_mask|) Σ CE, mean over batch.
                Numerical safeguards (short sequences + Bernoulli masks):
                  • t ∈ [t_eps, 1−t_eps] so mask mass is never almost always 0
                  • at least one masked eligible token per sequence when possible
                  • clamp (L/|𝒮_mask|) to max_diffusion_scale (paper ratio, capped)
                  • loss terms computed in float32 (autocast-safe)

  ℒ_UPS       = mean BCE over all B·L positions (prompt y = 1).

  ℒ           = ℒ_diffusion + λ_UPS · ℒ_UPS
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Avoid ρ_mask≈0 (almost no masks) and ρ_mask≈1 (degenerate UPS labels); still in (0,1).
_DEFAULT_T_EPS = 0.04


def remedi_training_step(
    model,                              # RemeDiMDMTransformer
    x0: torch.Tensor,                   # [B, L]  ground-truth tokens
    prompt_mask: torch.Tensor,          # [B, L]  bool; True = prompt (never masked)
    mask_id: int,
    vocab_size: int | None = None,      # for sampling random replacement tokens
    incorrect_ratio: float = 0.1,       # r in the paper
    lambda_ups: float = 0.3,            # λ_UPS from the paper
    t_eps: float = _DEFAULT_T_EPS,
    max_diffusion_scale: float | None = None,
    detach_tps: bool = False,           # (non-paper) block L_ups gradients from reaching TPS
    skip_diffusion: bool = False,       # (non-paper) zero out L_diffusion (TPS gets no gradient)
) -> dict[str, torch.Tensor]:
    """
    Compute the RemeDi SFT loss for one batch.

    Returns a dict with keys:
      "loss"            – ℒ_diffusion + λ_UPS · ℒ_UPS
      "loss_diffusion"  – scaled TPS CE on masked positions (Alg. 1)
      "loss_ups"        – mean UPS BCE over all positions (Alg. 1)
    """
    device = x0.device
    B, L   = x0.shape
    if vocab_size is None:
        vocab_size = int(model.config.vocab_size)
    cap = float(max_diffusion_scale) if max_diffusion_scale is not None else min(
        8.0 * math.sqrt(float(L)), 64.0
    )

    # ------------------------------------------------------------------
    # 1. Sample t in [t_eps, 1−t_eps] and noise ratios (stable Bernoulli mass)
    # ------------------------------------------------------------------
    t          = torch.rand(B, device=device).mul_(1.0 - 2.0 * t_eps).add_(t_eps)
    rho_mask   = t
    rho_incorr = 4.0 * incorrect_ratio * t * (1.0 - t)

    eligible = ~prompt_mask

    # ------------------------------------------------------------------
    # 2. Bernoulli mask, then Bernoulli incorrect on remaining (Alg. 1)
    # ------------------------------------------------------------------
    u_mask = torch.rand(B, L, device=device)
    mask_indices = eligible & (u_mask < rho_mask.unsqueeze(1))

    u_incorr = torch.rand(B, L, device=device)
    incorr_indices = eligible & (~mask_indices) & (u_incorr < rho_incorr.unsqueeze(1))

    # Guarantee ≥1 masked token when the answer region is non-empty (Alg. 1 assumes S_mask used).
    n_m = mask_indices.sum(dim=1)
    need = (n_m == 0) & eligible.any(dim=1)
    if need.any():
        b_fix = need.nonzero(as_tuple=True)[0]
        for b in b_fix:
            el_ix = eligible[b].nonzero(as_tuple=True)[0]
            pick = el_ix[torch.randint(len(el_ix), (1,), device=device)]
            mask_indices[b, pick] = True
            incorr_indices[b, pick] = False

    # ------------------------------------------------------------------
    # 3. Random wrong tokens (≠ mask_id, ≠ x0)
    # ------------------------------------------------------------------
    rand_tokens = torch.randint(0, vocab_size, (B, L), device=device)
    invalid = (rand_tokens == mask_id) | (rand_tokens == x0)
    while invalid.any():
        rand_tokens[invalid] = torch.randint(0, vocab_size, (invalid.sum().item(),), device=device)
        invalid = (rand_tokens == mask_id) | (rand_tokens == x0)

    x_s = x0.clone()
    x_s = torch.where(mask_indices, torch.full_like(x_s, mask_id), x_s)
    x_s = torch.where(incorr_indices, rand_tokens, x_s)

    # ------------------------------------------------------------------
    # 4. Forward on x_s → TPS logits + UPS confidences
    # ------------------------------------------------------------------
    logits_s, confidences = model.forward_with_confidence(x_s, detach_tps=detach_tps)
    logits_s = logits_s.float()
    confidences = confidences.float()

    # ------------------------------------------------------------------
    # 5. ℒ_diffusion — Algorithm 1 line 8: (L/|𝒮_mask|) Σ CE, scale clamped
    # ------------------------------------------------------------------
    ce_pos = F.cross_entropy(
        logits_s.reshape(B * L, -1),
        x0.reshape(-1),
        reduction="none",
    ).reshape(B, L)
    n_mask = mask_indices.sum(dim=1).clamp(min=0)
    sum_ce = (ce_pos * mask_indices.float()).sum(dim=1)
    scale = (float(L) / n_mask.float().clamp(min=1.0)).clamp(max=cap)
    per_seq = sum_ce * scale
    loss_diffusion = per_seq.mean()

    # ------------------------------------------------------------------
    # 6. ℒ_UPS — Algorithm 1 lines 9–15: mean over all L positions
    # ------------------------------------------------------------------
    with torch.no_grad():
        soft = F.softmax(logits_s, dim=-1)
        prob_correct = soft.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    ups_labels = torch.ones(B, L, device=device, dtype=prob_correct.dtype)
    ups_labels = ups_labels.masked_fill(incorr_indices, 0.0)
    ups_labels = torch.where(mask_indices, prob_correct, ups_labels)

    loss_ups = F.binary_cross_entropy_with_logits(
        confidences, ups_labels, reduction="mean",
    )

    # ------------------------------------------------------------------
    # 7. Combined loss
    # ------------------------------------------------------------------
    loss = (torch.zeros_like(loss_diffusion) if skip_diffusion else loss_diffusion) + lambda_ups * loss_ups

    return {
        "loss":           loss,
        "loss_diffusion": loss_diffusion,
        "loss_ups":       loss_ups,
    }
