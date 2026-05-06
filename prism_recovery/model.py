"""
PrismMDMTransformer
===================
Extends MDMTransformer with a lightweight per-token adapter head used for
PRISM self-correction training and inference.

Only new code lives here; the backbone is unchanged so pretrained
MDMTransformer weights load directly via load_state_dict(strict=False).
"""

import torch
import torch.nn as nn
from model.transformer import MDMTransformer, MDMConfig


class AdapterHead(nn.Module):
    """
    Predicts per-token correctness probability (binary).

    Input : backbone hidden states  [B, L, hidden_size]
    Output: per-token logits        [B, L]   (apply sigmoid → P(correct))

    A sigmoid value near 1 means "this token is likely correct";
    near 0 means "this token may need to be remasked."
    """
    def __init__(self, hidden_size: int, adapter_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, hidden_size]
        return self.net(hidden_states).squeeze(-1)   # [B, L]


class PrismMDMTransformer(MDMTransformer):
    """
    MDMTransformer + AdapterHead.

    Public API additions:
      forward_with_hidden(input_ids) → (logits, hidden_states)
          Returns the LM logits AND the pre-LM-head hidden states.
          Use this inside PRISM training/sampling.

      adapter: AdapterHead
          Per-token binary quality head.  Trained during PRISM fine-tuning;
          backbone parameters are frozen.

      forward(input_ids) → logits
          Identical to MDMTransformer.forward() — keeps compatibility with
          all existing code paths (mdm_sampling, evaluate_ddp_sudoku, …).
    """

    def __init__(self, config: MDMConfig, adapter_hidden: int = 64):
        super().__init__(config)
        self.adapter = AdapterHead(config.hidden_size, adapter_hidden)

    # ------------------------------------------------------------------
    # New method: returns both logits and the hidden states
    # ------------------------------------------------------------------
    def forward_with_hidden(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits        : [B, L, vocab_size]
        hidden_states : [B, L, hidden_size]  (after final_norm, before lm_head)
        """
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        hidden = self.final_norm(x)        # [B, L, hidden_size]
        logits = self.lm_head(hidden)      # [B, L, vocab_size]
        return logits, hidden

    # ------------------------------------------------------------------
    # Standard forward — identical interface to MDMTransformer
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_hidden(input_ids)
        return logits
