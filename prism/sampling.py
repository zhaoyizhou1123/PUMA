"""
PRISM inference
===============
Two-phase sampling that combines standard MDM decoding with
adapter-guided remasking of low-confidence already-decoded tokens.

Phase 1 (all steps):  standard top-k confidence unmasking (identical to mdm_sampling)
Phase 2 (steps in [step_on, step_off)): after each unmask, the adapter
    scores every already-decoded non-prompt token and remasks the
    num_remask lowest-confidence ones, giving the model a chance to
    correct them in subsequent steps.
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
    confidence    : str    – scoring strategy ("top_k" | "top_k_margin")
    unmasking_num : int    – tokens to unmask per step
    step_on       : int    – first step at which to apply remasking  (default 0)
    step_off      : int    – step at which to stop remasking  (default ∞)
    num_remask    : int    – clean tokens to remask per step  (default 1)

    Returns
    -------
    xt  [B, L] fully decoded sequence, or (xt, track_xt) if track=True.
    """
    temperature   = sampling_cfg.temperature
    confidence    = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    step_on       = getattr(sampling_cfg, "step_on",      0)
    step_off_val  = getattr(sampling_cfg, "step_off",     float('inf'))
    num_remask    = getattr(sampling_cfg, "num_remask",   0)
    unmask_mode   = getattr(sampling_cfg, "unmask_mode",  "random")

    if prompt_mask is None:
        prompt_mask = torch.zeros_like(xt, dtype=torch.bool)

    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_list = []

    # Net tokens decoded per step = unmasking_num (remasking is compensated by
    # revealing unmasking_num + num_remask tokens when remasking is active).
    # Loop bound is based on net progress = unmasking_num per step.
    n_answer = (~prompt_mask[0]).sum().item()
    max_steps = math.ceil(n_answer / unmasking_num) + 1

    for step in range(max_steps):
        mask_indices = (xt == mask_id)
        if mask_indices.sum() == 0:
            break

        # ---- backbone forward on current state xt ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits, _ = model.forward_with_hidden(xt)          # [B,L,V]

        logits_noisy = gumbel_softmax(logits, temperature=temperature)
        p = F.softmax(logits, dim=-1)

        # ---- unmasking order (random or top_k confidence) ----
        if unmask_mode == "top_k":
            unmask_score = torch.where(mask_indices, p.max(dim=-1).values,
                                       torch.full((B, L), float('-inf'), device=xt.device))
        else:  # random
            unmask_score = torch.where(
                mask_indices,
                torch.rand(B, L, device=xt.device),
                torch.full((B, L), float('-inf'), device=xt.device),
            )

        # ---- adapter-guided remasking (PRISM phase) ----
        # Per Algorithm 3 of the paper: unmask (unmasking_num + num_remask) tokens,
        # then remask num_remask low-confidence clean tokens.
        # Net progress = unmasking_num per step regardless of num_remask.
        #
        # Edge case: when fewer masks remain than (unmasking_num + num_remask),
        # unmask all remaining and skip remasking — otherwise we deadlock
        # (unmask N == remask N → zero net progress forever).
        remasking_active = (step_on <= step < step_off_val) and num_remask > 0

        per_seq_masked   = mask_indices.sum(dim=-1)                        # [B]
        n_unmask_target  = unmasking_num + (num_remask if remasking_active else 0)
        enough           = per_seq_masked >= n_unmask_target               # [B]
        per_seq_n_unmask = torch.where(enough,
            torch.full_like(per_seq_masked, n_unmask_target),
            per_seq_masked)                                                # [B]
        per_seq_n_remask = torch.where(enough,
            torch.full_like(per_seq_masked, num_remask if remasking_active else 0),
            torch.zeros_like(per_seq_masked))                             # [B]

        k_max = int(per_seq_n_unmask.max().item())
        if k_max > 0:
            _, sel  = unmask_score.topk(k=k_max, dim=-1)                  # [B, k_max]
            valid   = mask_indices.gather(1, sel)
            k_range = torch.arange(k_max, device=xt.device).unsqueeze(0)
            in_bud  = k_range < per_seq_n_unmask.unsqueeze(1)
            upd     = torch.zeros_like(mask_indices)
            upd.scatter_(1, sel, valid & in_bud)
            new_tok = logits_noisy.argmax(dim=-1)
            xt      = torch.where(upd, new_tok, xt)

        if remasking_active and per_seq_n_remask.sum() > 0:
            clean        = (~(xt == mask_id)) & (~prompt_mask)
            # PRISM scores the post-unmask sequence, so recompute hidden states
            # after the newly selected tokens have been filled in.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                _, hidden_post = model.forward_with_hidden(xt)             # [B,L,H]
            adapter_logits = model.adapter(hidden_post)                    # [B, L]
            remask_score = torch.where(clean, -adapter_logits,
                torch.full_like(adapter_logits, float('-inf')))
            k_rm = int(per_seq_n_remask.max().item())
            if k_rm > 0:
                _, rm_idx  = remask_score.topk(k=k_rm, dim=-1)
                rm_valid   = clean.gather(1, rm_idx)
                k_range_rm = torch.arange(k_rm, device=xt.device).unsqueeze(0)
                in_bud_rm  = k_range_rm < per_seq_n_remask.unsqueeze(1)
                rm_mask    = torch.zeros_like(clean)
                rm_mask.scatter_(1, rm_idx, rm_valid & in_bud_rm)
                xt = torch.where(rm_mask, torch.full_like(xt, mask_id), xt)

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
