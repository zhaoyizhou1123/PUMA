"""
PRISM fine-tuning algorithm
============================
Reference: "PRISM: Provable Self-Correction via Masked Diffusion" (arXiv 2510.01384)

Per-batch training step:
  1. Randomly mask x0 → x_t  (uniform masking ratio, skip prompt positions)
  2. Backbone forward (frozen by default) → logits, hidden_states
  3. One-step unmask: greedily reveal the top-k most confident masked tokens → x_s
  4. Binary labels: 1 if x_s[i] == x0[i] (correctly predicted), else 0
  5. Self-correction loss: BCE on the positions that were updated in step 3
  6. (Optional) regularisation loss: standard CE on the backbone predictions

The caller is responsible for freezing backbone parameters before passing
the model here.  See prism/train.py for the recommended usage pattern.
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
    reg_lambda: float = 0.5,            # weight on backbone CE regularisation loss
    tune_backbone: bool = False,        # if True, gradients flow through backbone
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
    # 1. Random masking:  sample how many tokens to mask per sequence,
    #    then pick that many random non-prompt positions.
    # ------------------------------------------------------------------
    L_eff = (~prompt_mask).sum(dim=1).float()                         # [B]
    num_mask = (torch.rand(B, device=device) * L_eff).long().clamp(min=1)  # [B]

    # Assign +inf score to prompt positions so they sort to the end
    noise = torch.rand(B, L, device=device).masked_fill(prompt_mask, float('inf'))
    order = noise.argsort(dim=1)
    mask_indices = (order < num_mask.unsqueeze(1))   # [B, L] bool
    x_t = torch.where(mask_indices, mask_id, x0)

    # ------------------------------------------------------------------
    # 2. Backbone forward
    #    No grad by default (backbone frozen); enable if tune_backbone=True
    # ------------------------------------------------------------------
    if tune_backbone:
        logits, hidden = model.forward_with_hidden(x_t)
    else:
        with torch.no_grad():
            logits, hidden = model.forward_with_hidden(x_t)
        hidden = hidden.detach()    # stop grad at adapter input

    # ------------------------------------------------------------------
    # 3. One-step unmask:  pick the top-k most confident masked tokens,
    #    assign them the backbone's argmax prediction → x_s
    # ------------------------------------------------------------------
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)                           # [B, L, V]
        conf = p.max(dim=-1).values                                 # [B, L]
        unmask_score = torch.where(
            mask_indices, conf, torch.full_like(conf, float('-inf'))
        )

        k = min(num_demasking_tokens, int(mask_indices.sum(dim=-1).max().item()))
        update_mask = torch.zeros_like(mask_indices)               # [B, L] bool

        if k > 0:
            _, topk_idx = unmask_score.topk(k=k, dim=-1)          # [B, k]
            valid = mask_indices.gather(1, topk_idx)               # [B, k]
            update_mask.scatter_(1, topk_idx, valid)

            new_tokens = logits.argmax(dim=-1)                     # [B, L]
            x_s = torch.where(update_mask, new_tokens, x_t)
        else:
            x_s = x_t

    # ------------------------------------------------------------------
    # 4. Binary labels:  1 = correctly decoded, 0 = wrong
    # ------------------------------------------------------------------
    binary_labels = (x_s == x0).float()                            # [B, L]

    # ------------------------------------------------------------------
    # 5. Self-correction loss (adapter head trained here)
    # ------------------------------------------------------------------
    n_updated = update_mask.sum().item()
    if n_updated == 0:
        # Degenerate batch — keep computation graph alive but zero loss
        adapter_logits = model.adapter(hidden)
        loss_sc = adapter_logits.sum() * 0.0
    else:
        adapter_logits = model.adapter(hidden)                     # [B, L]
        loss_sc = F.binary_cross_entropy_with_logits(
            adapter_logits[update_mask],
            binary_labels[update_mask],
        )

    # ------------------------------------------------------------------
    # 6. (Optional) backbone regularisation — CE on masked positions
    # ------------------------------------------------------------------
    result: dict[str, torch.Tensor] = {"loss_sc": loss_sc}

    if reg_lambda > 0.0 and mask_indices.sum().item() > 0:
        loss_reg = F.cross_entropy(
            logits[mask_indices],
            x0[mask_indices],
        )
        loss = loss_sc + reg_lambda * loss_reg
        result["loss_reg"] = loss_reg
    else:
        loss = loss_sc

    result["loss"] = loss
    return result
