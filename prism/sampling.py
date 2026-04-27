"""
PRISM inference
===============
Matches the official implementation (PRISM_llada/sampling.py):

Per step:
  1. Single forward pass on current x → logits + hidden states
  2. Remask: score clean non-prompt tokens with -adapter_conf from this forward;
     select num_remask lowest-confidence tokens and mask them back;
     increase unmask budget by the number actually remasked
  3. Unmask: from the now-larger masked pool, select top unmasking_num tokens
     using random scores (official default)

NFE = 1 per step.
"""

from __future__ import annotations
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from sampling import gumbel_softmax     # reuse existing helper


@torch.no_grad()
def prism_sampling(
    model,                              # PrismMDMTransformer
    xt: torch.Tensor,                   # [B, L]  initial sequence (response = mask_id)
    mask_id: int,
    sampling_cfg,
    prompt_mask: torch.Tensor = None,   # [B, L] bool; True = prompt
    device: torch.device = None,
    track: bool = False,
) -> torch.Tensor:
    """
    Parameters in sampling_cfg
    --------------------------
    temperature   : float  – Gumbel-softmax temperature (0 = argmax)
    unmasking_num : int    – net tokens to unmask per step
    step_on       : int    – first step at which to apply remasking  (default 0)
    step_off      : int    – step at which to stop remasking  (default ∞)
    num_remask    : int    – clean tokens to remask per step  (default 0)

    Returns
    -------
    xt  [B, L] fully decoded sequence, or (xt, track_xt) if track=True.
    """
    temperature   = sampling_cfg.temperature
    unmasking_num = sampling_cfg.unmasking_num
    step_on       = getattr(sampling_cfg, "step_on",    0)
    step_off_val  = getattr(sampling_cfg, "step_off",   float('inf'))
    num_remask    = getattr(sampling_cfg, "num_remask",  0)

    if prompt_mask is None:
        prompt_mask = torch.zeros_like(xt, dtype=torch.bool)

    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_list = []

    n_answer  = (~prompt_mask[0]).sum().item()
    max_steps = math.ceil(n_answer / unmasking_num) + 1

    for step in range(max_steps):
        mask_indices = (xt == mask_id)
        if mask_indices.sum() == 0:
            break

        # ---- single forward pass ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits, hidden = model.forward_with_hidden(xt)     # [B,L,V], [B,L,H]

        remasking_active = (step_on <= step < step_off_val) and num_remask > 0

        # ---- remasking: score clean tokens with adapter, mask back num_remask lowest ----
        # Increases unmask budget by however many tokens were actually remasked,
        # so net progress = unmasking_num per step (matches official sampling.py:184).
        per_seq_n_remasked = torch.zeros(B, dtype=torch.long, device=xt.device)
        if remasking_active:
            clean = (~mask_indices) & (~prompt_mask)            # [B, L]
            adapter_logits = model.adapter(hidden)              # [B, L]
            remask_score = torch.where(clean, -adapter_logits, torch.full((B, L), float('-inf'), device=xt.device))
            for j in range(B):
                n_clean = clean[j].sum().item()
                k = min(num_remask, int(n_clean))
                if k > 0:
                    _, idx = remask_score[j].topk(k)
                    xt[j].index_fill_(0, idx, mask_id)
                    per_seq_n_remasked[j] = k

        if track:
            track_list.append(xt.clone().detach().cpu())

        # ---- unmasking: random scores on masked positions (official default) ----
        mask_indices = (xt == mask_id)                          # recompute after remasking
        logits_noisy = gumbel_softmax(logits, temperature=temperature)
        new_tok      = logits_noisy.argmax(dim=-1)              # [B, L]

        unmask_score = torch.where(
            mask_indices,
            torch.rand(B, L, device=xt.device),
            torch.full((B, L), float('-inf'), device=xt.device),
        )

        # budget = unmasking_num + however many were just remasked (per official line 184)
        per_seq_budget = torch.full((B,), unmasking_num, dtype=torch.long, device=xt.device)
        per_seq_budget = per_seq_budget + per_seq_n_remasked
        per_seq_budget = torch.minimum(per_seq_budget, mask_indices.sum(dim=-1))

        k_max = int(per_seq_budget.max().item())
        if k_max > 0:
            _, sel  = unmask_score.topk(k=k_max, dim=-1)
            valid   = mask_indices.gather(1, sel)
            k_range = torch.arange(k_max, device=xt.device).unsqueeze(0)
            in_bud  = k_range < per_seq_budget.unsqueeze(1)
            upd     = torch.zeros_like(mask_indices)
            upd.scatter_(1, sel, valid & in_bud)
            xt      = torch.where(upd, new_tok, xt)

        if track:
            track_list.append(xt.clone().detach().cpu())

    if track:
        return xt, torch.stack(track_list, dim=0)   # (T, B, L)
    return xt


# ---------------------------------------------------------------------------
# Evaluation helper — mirrors evaluate_ddp_sudoku but uses prism_sampling
# ---------------------------------------------------------------------------

def prism_evaluate_ddp_sudoku(model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None):
    """
    Drop-in replacement for evaluate_ddp_sudoku that uses prism_sampling.
    Evaluates solve accuracy with adapter-guided remasking.
    """
    from eval.sudoku_eval import verify_sudoku

    val_dir  = cfg.validation.val_dir
    mask_id  = cfg.data.mask_id
    seq_len  = cfg.model.max_position
    n4       = seq_len // 2
    n        = round(n4 ** 0.25)

    test_mdm_path = os.path.join(val_dir, f"test_mdm_n{n}.npy")
    if not os.path.exists(test_mdm_path):
        test_mdm_path = os.path.join(val_dir, "test_mdm.npy")
    if not os.path.exists(test_mdm_path):
        raise FileNotFoundError(f"No test_mdm_n{n}.npy or test_mdm.npy in {val_dir}")

    X = np.load(test_mdm_path).copy()
    Y = np.load(test_mdm_path)
    X[:, n4:] = mask_id

    N     = len(X)
    ratio = cfg.validation.ratio
    N_val = int(N * ratio)
    X, Y  = X[:N_val], Y[:N_val]

    per_rank  = math.ceil(N_val / world_size)
    start     = rank * per_rank
    end       = min(start + per_rank, N_val)
    batch_size = cfg.validation.get("batch_size", 64)
    num_batches = math.ceil((end - start) / batch_size)

    local_correct, local_total = 0, 0
    with torch.no_grad():
        for j in tqdm(range(num_batches), desc="PRISM Eval", disable=(rank != 0)):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X = torch.from_numpy(X[s:e]).long().to(device)
            batch_Y = torch.from_numpy(Y[s:e]).long().to(device)
            prompt_mask = torch.zeros_like(batch_X, dtype=torch.bool)
            prompt_mask[:, :n4] = True

            pred = prism_sampling(model, batch_X, mask_id, sampling, prompt_mask=prompt_mask, device=device)
            matches = verify_sudoku(pred, batch_Y, n)
            local_correct += matches.sum().item()
            local_total   += batch_Y.shape[0]

    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    g_correct, g_total = tensor.tolist()
    return g_correct / max(g_total, 1)
