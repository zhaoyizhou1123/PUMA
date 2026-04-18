"""
RemeDiMDMTransformer
====================
Extends MDMTransformer with a dual-stream architecture following the RemeDi paper:
  "Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models"
  arXiv:2509.23653

Architecture
------------
Token Prediction Stream (TPS): the original MDMTransformer forward pass.
Unmasking Policy Stream (UPS): taps into TPS at specific layer indices, accumulates
  hidden states, processes them through dedicated TransformerBlocks, back-projects
  the result residually into TPS, and finally produces per-token scalar confidence
  scores via a linear head.

UPS positions (default [0, 2, 5, 7] for an 8-layer model) correspond to roughly
0%, 25%, 63%, 88% depth — scaling the paper's [0, 10, 20, 30]/32 tap points
(the paper's 32-layer model uses 0-indexed positions 0,10,20,30).

At each tap i the forward pass does:
  1. upm_feat  ← upm_feat + x          (accumulate TPS hidden state)
  2. upm_out   ← ups_blocks[i](upm_feat)
  3. x         ← x + backprop_norms[i](backprop_linears[i](upm_out))   (write back)
  4. upm_feat  ← upm_out               (carry processed UPS state forward)

After the final transformer layer the TPS and UPS heads are computed separately:
  logits      = lm_head(final_norm(x))                         [B, L, V]
  confidences = confidence_head(ups_final_norm(upm_feat))      [B, L]

A high confidence score means "this token position is likely correct and should stay
decoded"; a low score means "remask me".

Pretrained MDMTransformer weights load cleanly via load_state_dict(strict=False).
All new parameters (ups_blocks, backprop_*, confidence_head, ups_final_norm) are
freshly initialised and trained during RemeDi SFT fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.transformer import MDMTransformer, MDMConfig, TransformerBlock
from model.utils import RMSNorm


class RemeDiMDMTransformer(MDMTransformer):
    """
    MDMTransformer + Unmasking Policy Stream (UPS).

    Public API additions
    --------------------
    forward_with_confidence(input_ids) → (logits, confidences)
        logits      : [B, L, vocab_size]  — TPS vocabulary predictions
        confidences : [B, L]              — UPS per-token quality scores (pre-sigmoid)

    forward(input_ids) → logits
        Identical to MDMTransformer.forward() — backward-compatible.
    """

    def __init__(
        self,
        config: MDMConfig,
        ups_positions: tuple[int, ...] = (0, 2, 5, 7),
    ):
        super().__init__(config)
        self.ups_positions = list(ups_positions)
        self._ups_pos_set  = set(ups_positions)
        n_ups = len(ups_positions)

        # One TransformerBlock per tap — processes accumulated UPS features
        self.ups_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(n_ups)
        ])

        # Back-projection from UPS → TPS (one per tap).
        # Zero-initialized per the paper: "zero-initialized projections to
        # preserve original TPS behavior at the start of fine-tuning."
        self.backprop_linears = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(n_ups)
        ])
        self.backprop_norms = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in range(n_ups)
        ])

        # UPS output head (small init — matches official RemeDi; default Linear is too hot for BCE)
        self.ups_final_norm   = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.confidence_head  = nn.Linear(config.hidden_size, 1, bias=False)
        nn.init.normal_(self.confidence_head.weight, mean=0.0, std=0.02)

        # Zero-init back-projection linears so initial UPS write-back is zero,
        # leaving the pretrained TPS unchanged at the start of fine-tuning.
        for linear in self.backprop_linears:
            nn.init.zeros_(linear.weight)

    # ------------------------------------------------------------------
    # Core forward with both TPS logits and UPS confidence scores
    # ------------------------------------------------------------------
    def forward_with_confidence(
        self, input_ids: torch.Tensor, detach_tps: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits      : [B, L, vocab_size]  TPS output
        confidences : [B, L]              UPS output (pre-sigmoid; higher = more confident)
        """
        x        = self.emb(input_ids)
        upm_feat = None
        ups_idx  = 0

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in self._ups_pos_set:
                # 1. Accumulate TPS hidden state into UPS stream
                x_for_ups = x.detach() if detach_tps else x
                upm_feat = x_for_ups if upm_feat is None else upm_feat + x_for_ups

                # 2. Process accumulated UPS features
                upm_out = self.ups_blocks[ups_idx](upm_feat)

                # 3. Back-project into TPS (residual write-back)
                #    Paper: linear(norm(upm_feat)) — norm first, then linear
                back = self.backprop_linears[ups_idx](
                    self.backprop_norms[ups_idx](upm_out)
                )
                x = x + back

                # 4. Carry processed UPS state forward
                upm_feat = upm_out

                ups_idx += 1

        # TPS head
        tps_hidden = self.final_norm(x)
        logits     = self.lm_head(tps_hidden)           # [B, L, V]

        # UPS head — use fallback if UPS was never triggered (shouldn't happen)
        if upm_feat is not None:
            ups_hidden  = self.ups_final_norm(upm_feat)
        else:
            ups_hidden  = tps_hidden
        confidences = self.confidence_head(ups_hidden).squeeze(-1)  # [B, L]

        return logits, confidences

    # ------------------------------------------------------------------
    # Standard forward — identical interface to MDMTransformer
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_confidence(input_ids)
        return logits
