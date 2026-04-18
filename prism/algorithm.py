"""
PRISM fine-tuning algorithm
============================
Reference: "PRISM: Provable Self-Correction via Masked Diffusion" (arXiv 2510.01384)

Per-batch training step:
  1. Sample t ~ Uniform(eps, 1); mask x0 → x_t via Bernoulli(move_chance=t) per token
  2. Backbone forward on x_t → logits  (used for regularisation loss)
  3. One-step unmask: randomly select k masked tokens → x_s  (uniform, not confidence-based)
  4. Backbone forward on x_s → hidden_s  (adapter sees post-unmask context)
  5. Binary labels: 1 if x_s[i] == x0[i] at updated positions, else 0
  6. Self-correction loss: BCE(adapter(hidden_s)[updated], labels[updated])
  7. (Optional) regularisation loss: CE(logits[masked], x0[masked]) × (1/t) × λ

Key design: adapter is trained on x_s (post-unmask hidden states), matching
inference where it scores already-decoded tokens in a partially filled sequence.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional


def prism_training_step(
    model,                              # PrismMDMTransformer (backbone already frozen)
    x0: torch.Tensor,                   # [B, L]  ground-truth tokens
    prompt_mask: torch.Tensor,          # [B, L]  bool; True = prompt (never masked)
    mask_id: int,
    num_demasking_tokens: int,          # k: how many tokens to reveal in one step
    reg_lambda: float = 5.0,            # weight on backbone CE regularisation loss
    tune_backbone: bool = False,        # if True, gradients flow through backbone
    sampling_eps: float = 0.001,        # lower bound for t sampling
    unmask_mode: str = "random",        # "random" or "top_k" for selecting which masked tokens to fill
) -> dict[str, torch.Tensor]:
    """
    Compute the PRISM fine-tuning loss for one batch.

    Returns a dict with keys:
      "loss"     – total loss (loss_sc + reg_lambda * loss_reg)
      "loss_sc"  – self-correction BCE loss
      "loss_reg" – backbone CE regularisation loss  (only if reg_lambda > 0)
    """
    device = x0.device
    B, L = x0.shape

    # ------------------------------------------------------------------
    # 1. Sample t ~ Uniform(eps, 1); mask via Bernoulli(move_chance=t)
    #    per non-prompt token (log-linear noise schedule: move_chance = t).
    #    Prompt positions are never masked.
    # ------------------------------------------------------------------
    t = torch.rand(B, device=device) * (1 - sampling_eps) + sampling_eps  # [B]
    move_chance = t[:, None]                                               # [B, 1]
    rand_mask = torch.rand(B, L, device=device) < move_chance
    mask_indices = rand_mask & ~prompt_mask                                # [B, L] bool
    x_t = torch.where(mask_indices, mask_id, x0)

    # ------------------------------------------------------------------
    # 2. Backbone forward on x_t (for regularisation loss)
    # ------------------------------------------------------------------
    if tune_backbone:
        logits, _ = model.forward_with_hidden(x_t)
    else:
        with torch.no_grad():
            logits, _ = model.forward_with_hidden(x_t)

    # ------------------------------------------------------------------
    # 3. One-step unmask: randomly select k masked tokens → x_s
    #    (uniform random, not confidence-based — per reference code)
    # ------------------------------------------------------------------
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)
        if unmask_mode == "top_k":
            conf = p.max(dim=-1).values
            unmask_score = torch.where(mask_indices, conf, torch.full((B, L), float('-inf'), device=device))
        else:  # random
            unmask_score = torch.where(
                mask_indices,
                torch.rand(B, L, device=device),
                torch.full((B, L), float('-inf'), device=device),
            )

        k = min(num_demasking_tokens, int(mask_indices.sum(dim=-1).max().item()))
        update_mask = torch.zeros_like(mask_indices)

        if k > 0:
            _, topk_idx = unmask_score.topk(k=k, dim=-1)
            valid = mask_indices.gather(1, topk_idx)
            update_mask.scatter_(1, topk_idx, valid)
            new_tokens = torch.multinomial(p.view(B * L, -1), num_samples=1).view(B, L)
            x_s = torch.where(update_mask, new_tokens, x_t)
        else:
            x_s = x_t

    # ------------------------------------------------------------------
    # 4. Backbone forward on x_s (post-unmask) for adapter supervision.
    #    Per the paper: adapter scores the sequence AFTER new tokens are
    #    filled in, so it sees full context including freshly decoded tokens.
    # ------------------------------------------------------------------
    if tune_backbone:
        _, hidden_s = model.forward_with_hidden(x_s)
    else:
        with torch.no_grad():
            _, hidden_s = model.forward_with_hidden(x_s)
        hidden_s = hidden_s.detach()

    # ------------------------------------------------------------------
    # 5. Binary labels: 1 = updated token is correct, 0 = wrong
    #    Only supervise on updated_indices = positions that changed xt→xs
    # ------------------------------------------------------------------
    updated_indices = update_mask  # positions that were just decoded
    binary_labels = (x_s == x0).float()

    # ------------------------------------------------------------------
    # 6. Self-correction loss (adapter scores x_s, supervised on updated positions)
    # ------------------------------------------------------------------
    n_updated = updated_indices.sum().item()
    if n_updated == 0:
        adapter_logits = model.adapter(hidden_s)
        loss_sc = adapter_logits.sum() * 0.0
    else:
        adapter_logits = model.adapter(hidden_s)                   # [B, L]
        loss_sc = F.binary_cross_entropy_with_logits(
            adapter_logits[updated_indices],
            binary_labels[updated_indices],
        )

    # ------------------------------------------------------------------
    # 7. (Optional) backbone regularisation — CE on masked positions,
    #    reweighted by 1/t (per reference loss.py AdapterFinetunePRISMLoss)
    # ------------------------------------------------------------------
    result: dict[str, torch.Tensor] = {"loss_sc": loss_sc}

    if reg_lambda > 0.0 and mask_indices.sum().item() > 0:
        mask_reweighting = (1.0 / t).unsqueeze(1).expand(B, L)    # [B, L]
        ce_loss = F.cross_entropy(
            logits[mask_indices],
            x0[mask_indices],
            reduction='none',
        )
        ce_loss = ce_loss * mask_reweighting[mask_indices]
        loss_reg = ce_loss.sum() / (B * L)
        loss = loss_sc + reg_lambda * loss_reg
        result["loss_reg"] = loss_reg
    else:
        loss = loss_sc

    result["loss"] = loss
    return result
