"""
BackPlayMDMTransformer
======================
Extends MDMTransformer with a per-token correction head used for
BackPlay (arXiv 2601.06428) self-correction training and inference.

BackPlay uses a shallow Transformer correction head on the backbone's
penultimate hidden states rather than an MLP probe.
"""

import torch
import torch.nn as nn
from model.transformer import MDMTransformer, MDMConfig, TransformerBlock
from model.utils import RMSNorm


class AdapterHead(nn.Module):
    """
    Predicts per-token correctness probability (binary).

    Input : backbone hidden states  [B, L, hidden_size]
    Output: per-token logits        [B, L]   (apply sigmoid -> P(correct))
    """

    def __init__(self, config: MDMConfig, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(max(1, num_layers))
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.out_proj(x).squeeze(-1)


class BackPlayMDMTransformer(MDMTransformer):
    """
    MDMTransformer + Transformer correction head for BackPlay.
    """

    def __init__(self, config: MDMConfig, adapter_layers: int = 2):
        super().__init__(config)
        self.adapter = AdapterHead(config, num_layers=adapter_layers)

    def forward_with_hidden(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits        : [B, L, vocab_size]
        hidden_states : [B, L, hidden_size]  (penultimate layer h^{L-1})
        """
        x = self.emb(input_ids)
        for layer in self.layers[:-1]:
            x = layer(x)
        hidden = x
        x = self.layers[-1](x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, hidden

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_hidden(input_ids)
        return logits
