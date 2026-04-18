"""
BackPlay (LBC) fine-tuning algorithm
=====================================
Reference: "BackPlay: Plug-in Look-Back Self-Correction for Diffusion
Language Models" (arXiv 2601.06428)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample_xt(
    x0: torch.Tensor,
    prompt_mask: torch.Tensor,
    t: torch.Tensor,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample x_t ~ q_{t|0}(.|x0) with independent masking on non-prompt tokens.
    """
    B, L = x0.shape
    rand = torch.rand(B, L, device=x0.device)
    mask_indices = (rand < t.unsqueeze(1)) & (~prompt_mask)
    x_t = torch.where(mask_indices, mask_id, x0)
    return x_t, mask_indices


def _sample_xt_plus_delta(
    x_t: torch.Tensor,
    prompt_mask: torch.Tensor,
    t: torch.Tensor,
    t_prime: torch.Tensor,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample x_{t+t'} ~ q_{t+t'|0,t}(.|x0, x_t) by additionally masking visible tokens.
    """
    B, L = x_t.shape
    rand = torch.rand(B, L, device=x_t.device)
    addl_mask_prob = (t_prime / (1.0 - t).clamp(min=1e-6)).clamp(max=1.0)
    eligible = (x_t != mask_id) & (~prompt_mask)
    extra_mask = (rand < addl_mask_prob.unsqueeze(1)) & eligible
    x_t_plus = torch.where(extra_mask, mask_id, x_t)
    mask_indices = (x_t_plus == mask_id) & (~prompt_mask)
    return x_t_plus, mask_indices


def backplay_training_step(
    model,
    x0: torch.Tensor,
    prompt_mask: torch.Tensor,
    mask_id: int,
    delta_t: float = 0.1,
    reg_lambda: float = 0.0,
    tune_backbone: bool = False,
    error_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Compute the BackPlay LBC loss for one batch following Algorithm 2.

    error_weight: multiplier on the loss for label=0 (error) positions.
      1.0 = vanilla BCE (paper default).
      >1.0 = up-weight the rare error class to counter class imbalance.
    """
    device = x0.device
    B, _ = x0.shape
    valid_positions = ~prompt_mask
    L_eff = valid_positions.sum(dim=1).float()

    # Sample t ~ Uniform[0, 1] and t' ~ Uniform[delta_t, 1-t] per Algorithm 2.
    # When 1-t < delta_t the t' range collapses to {delta_t}; clamp handles that.
    t = torch.rand(B, device=device)
    t_prime_max = (1.0 - t).clamp(min=delta_t)
    t_prime = delta_t + torch.rand(B, device=device) * (t_prime_max - delta_t)

    x_t, _ = _sample_xt(x0, prompt_mask, t, mask_id)
    x_tt, mask_indices_tt = _sample_xt_plus_delta(x_t, prompt_mask, t, t_prime, mask_id)

    if tune_backbone:
        logits, _ = model.forward_with_hidden(x_tt)
    else:
        with torch.no_grad():
            logits, _ = model.forward_with_hidden(x_tt)

    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        conf_masked = confidence.masked_fill(~mask_indices_tt, float("-inf"))

        k_per_seq = (L_eff * delta_t).ceil().long().clamp(min=1)
        num_masked = mask_indices_tt.sum(dim=1).long()
        k_per_seq = k_per_seq.clamp(max=num_masked)

        k_max = int(k_per_seq.max().item()) if k_per_seq.numel() > 0 else 0
        if k_max > 0:
            _, topk_idx = conf_masked.topk(k=k_max, dim=-1)
            rank = torch.arange(k_max, device=device).unsqueeze(0)
            valid_rank = rank < k_per_seq.unsqueeze(1)
            selected_masked = mask_indices_tt.gather(1, topk_idx)
            M = torch.zeros_like(mask_indices_tt)
            M.scatter_(1, topk_idx, valid_rank & selected_masked)
        else:
            M = torch.zeros_like(mask_indices_tt)

        # Algorithm 2 uses sampled artifact tokens y ~ p_theta(x0 | x_{t+t'}).
        sampled_tokens = torch.distributions.Categorical(logits=logits).sample()
        z_t = torch.where(M, sampled_tokens, x_t)

    if tune_backbone:
        _, hidden_s = model.forward_with_hidden(z_t)
    else:
        with torch.no_grad():
            _, hidden_s = model.forward_with_hidden(z_t)
        hidden_s = hidden_s.detach()

    binary_labels = (z_t == x0).float()
    adapter_logits = model.adapter(hidden_s)

    if valid_positions.sum().item() == 0:
        loss_sc = adapter_logits.sum() * 0.0
    else:
        labels_flat = binary_labels[valid_positions]
        if error_weight != 1.0:
            # Up-weight label=0 (error) positions to counter class imbalance.
            sample_weight = torch.where(labels_flat == 0,
                                        torch.tensor(error_weight, device=labels_flat.device),
                                        torch.ones_like(labels_flat))
        else:
            sample_weight = None
        loss_sc = F.binary_cross_entropy_with_logits(
            adapter_logits[valid_positions],
            labels_flat,
            weight=sample_weight,
        )

    result: dict[str, torch.Tensor] = {"loss_sc": loss_sc}
    if reg_lambda > 0.0 and mask_indices_tt.sum().item() > 0:
        loss_reg = F.cross_entropy(logits[mask_indices_tt], x0[mask_indices_tt])
        result["loss_reg"] = loss_reg
        result["loss"] = loss_sc + reg_lambda * loss_reg
    else:
        result["loss"] = loss_sc
    return result
