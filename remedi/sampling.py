"""
RemeDi inference — global top-K remasking
==========================================
Reference: arXiv:2509.23653  (inference.py in the official repo)

Core idea
---------
At each decoding step, rather than *monotonically* adding new tokens to the
decoded set, RemeDi scores **all** positions (already-decoded + newly predicted
masked positions) and keeps only the globally top-K most confident ones.
Previously decoded tokens with low UPS confidence can be evicted back to [MASK].

The number of committed (decoded) tokens grows by `unmasking_num` per step, so
total decoding steps ≈ n_answer / unmasking_num — same wall-clock as standard MDM.

Scoring strategy
----------------
  use_ups=False (default, matches maple-research-lab/RemeDi inference.py):
      per-position score = logits[i, x0_hat_i] with x0_hat = argmax logits
      (same token selection as the official script: masked → argmax, else keep)
  use_ups=True:
      per-position score = UPS pre-sigmoid confidence (paper prose)

Token prediction matches the official repo: greedy argmax on logits at masked
positions (temperature on the config is ignored for RemeDi path).

Prompt positions are pinned (score = +inf) and never remasked.

Sampling config fields (beyond standard ones)
---------------------------------------------
  unmasking_num  : int  — tokens committed per step (net progress)
  use_ups        : bool — default False (GitHub); True for paper-style UPS ranking
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm


@torch.no_grad()
def remedi_sampling(
    model,                              # RemeDiMDMTransformer
    xt: torch.Tensor,                   # [B, L]  initial sequence (answer = mask_id)
    mask_id: int,
    sampling_cfg,
    prompt_mask: torch.Tensor = None,   # [B, L] bool; True = prompt
    device: torch.device = None,
) -> torch.Tensor:
    """
    Global top-K remasking sampler.

    At step s, the model is allowed to keep `num_committed[s]` non-prompt
    tokens decoded (up from `num_committed[s-1]`).  The UPS/TPS score
    determines *which* tokens to keep — so low-confidence earlier decisions
    can be overridden by higher-confidence new predictions.

    Returns
    -------
    xt  [B, L] fully decoded sequence.
    """
    unmasking_num = sampling_cfg.unmasking_num
    use_ups       = getattr(sampling_cfg, "use_ups", False)

    if prompt_mask is None:
        prompt_mask = torch.zeros_like(xt, dtype=torch.bool)

    B, L = xt.shape
    xt   = xt.clone()

    n_answer  = int((~prompt_mask[0]).sum().item())   # non-prompt positions per seq
    max_steps = math.ceil(n_answer / unmasking_num) + 1

    for step in range(max_steps):
        mask_indices = xt == mask_id
        if mask_indices.sum() == 0:
            break

        # ---- forward: TPS logits + UPS confidences ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            logits, ups_conf = model.forward_with_confidence(xt)   # [B,L,V], [B,L]
            logits = logits.float()  # match official inference.py

        # ---- predict masked positions (greedy argmax — official RemeDi repo) ----
        x0_pred = logits.argmax(dim=-1)
        x_full  = torch.where(mask_indices, x0_pred, xt)

        # ---- ranking scores: TPS logit at chosen token vs UPS ----
        if use_ups:
            scores = ups_conf
        else:
            scores = logits.gather(dim=-1, index=x_full.unsqueeze(-1)).squeeze(-1)

        # Prompt positions are always kept
        scores = scores.masked_fill(prompt_mask, float('inf'))

        # ---- global top-K: keep num_committed non-prompt positions ----
        num_committed = min((step + 1) * unmasking_num, n_answer)

        # Work only on non-prompt positions for top-K selection
        non_prompt_scores = scores.masked_fill(prompt_mask, float('-inf'))
        # clamp k to the actual number of non-prompt positions
        k = min(num_committed, n_answer)
        _, topk_idx = non_prompt_scores.topk(k=k, dim=-1)           # [B, k]

        # Build new xt: start all non-prompt as masked, then fill top-K
        keep_mask = torch.zeros(B, L, dtype=torch.bool, device=xt.device)
        keep_mask.scatter_(1, topk_idx, True)
        keep_mask = keep_mask | prompt_mask                          # always keep prompt

        xt = torch.where(keep_mask, x_full, torch.full_like(x_full, mask_id))

    return xt


# ---------------------------------------------------------------------------
# Evaluation helper — mirrors prism_evaluate_ddp_sudoku
# ---------------------------------------------------------------------------

def remedi_evaluate_ddp_sudoku(
    model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None
):
    """
    Drop-in replacement for evaluate_ddp_sudoku that uses remedi_sampling.
    Evaluates solve accuracy with global top-K remasking.
    """
    from eval.sudoku_eval import verify_sudoku

    val_dir = cfg.validation.val_dir
    mask_id = cfg.data.mask_id
    seq_len = cfg.model.max_position
    n4      = seq_len // 2
    n       = round(n4 ** 0.25)

    test_mdm_path = os.path.join(val_dir, f"test_mdm_n{n}.npy")
    if not os.path.exists(test_mdm_path):
        test_mdm_path = os.path.join(val_dir, "test_mdm.npy")
    if not os.path.exists(test_mdm_path):
        raise FileNotFoundError(
            f"No test_mdm_n{n}.npy or test_mdm.npy in {val_dir}"
        )

    X = np.load(test_mdm_path).copy()
    Y = np.load(test_mdm_path)
    X[:, n4:] = mask_id

    N     = len(X)
    ratio = cfg.validation.ratio
    N_val = int(N * ratio)
    X, Y  = X[:N_val], Y[:N_val]

    per_rank    = math.ceil(N_val / world_size)
    start       = rank * per_rank
    end         = min(start + per_rank, N_val)
    batch_size  = cfg.validation.get("batch_size", 64)
    num_batches = math.ceil((end - start) / batch_size)

    local_correct, local_total = 0, 0
    with torch.no_grad():
        for j in tqdm(range(num_batches), desc="RemeDi Eval", disable=(rank != 0)):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X      = torch.from_numpy(X[s:e]).long().to(device)
            batch_Y      = torch.from_numpy(Y[s:e]).long().to(device)
            prompt_mask  = torch.zeros_like(batch_X, dtype=torch.bool)
            prompt_mask[:, :n4] = True

            pred    = remedi_sampling(
                model, batch_X, mask_id, sampling,
                prompt_mask=prompt_mask, device=device,
            )
            matches = verify_sudoku(pred, batch_Y, n)
            local_correct += matches.sum().item()
            local_total   += batch_Y.shape[0]

    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    g_correct, g_total = tensor.tolist()
    return g_correct / max(g_total, 1)
