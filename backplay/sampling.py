"""
BackPlay inference
==================
Dynamic self-correction inference from Algorithm 1 of the BackPlay paper:
confidence-based unmasking, top-K remasking, rollback-aware scheduling,
and an anti-repetitive FIFO block buffer.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from sampling import gumbel_softmax


@torch.no_grad()
def backplay_sampling(
    model,
    xt: torch.Tensor,
    mask_id: int,
    sampling_cfg,
    prompt_mask: torch.Tensor = None,
    device: torch.device = None,
    track: bool = False,
    return_stats: bool = False,
) -> torch.Tensor:
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    tau = getattr(sampling_cfg, "tau", 0.5)
    stride = getattr(sampling_cfg, "stride", 1)
    block_size = getattr(sampling_cfg, "block_size", 10)
    remask_budget = getattr(sampling_cfg, "remask_budget", 2)

    if prompt_mask is None:
        prompt_mask = torch.zeros_like(xt, dtype=torch.bool)

    B, _ = xt.shape
    xt = xt.clone()
    if track:
        track_list = []
    stats = {
        "num_steps": 0.0,
        "correction_steps": 0.0,
        "sequences_with_any_remask": 0.0,
        "total_remasked_tokens": 0.0,
        "score_count": 0.0,
        "error_prob_sum": 0.0,
        "error_prob_max_sum": 0.0,
        "error_prob_max_count": 0.0,
    }

    hblock = [[] for _ in range(B)]
    any_remask = torch.zeros(B, device=xt.device, dtype=torch.bool)
    answer_len = int((~prompt_mask[0]).sum().item())
    delta_t = max(unmasking_num, 1) / max(answer_len, 1)
    t = 1.0
    step = 0
    max_steps = 10 * math.ceil(answer_len / max(unmasking_num, 1)) + 10

    while t > 0 and step < max_steps:
        stats["num_steps"] += 1
        mask_indices = (xt == mask_id)
        if mask_indices.sum() == 0:
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits, _ = model.forward_with_hidden(xt)

        logits_noisy = gumbel_softmax(logits, temperature=temperature)
        probs = F.softmax(logits, dim=-1)

        if confidence == "top_k":
            unmask_score = probs.max(dim=-1).values
        elif confidence == "top_k_margin":
            top2 = probs.topk(k=2, dim=-1).values
            unmask_score = top2[..., 0] - top2[..., 1]
        else:
            unmask_score = probs.max(dim=-1).values
        unmask_score = unmask_score.masked_fill(~mask_indices, float("-inf"))

        xt_new = xt.clone()
        k_max = min(unmasking_num, int(mask_indices.sum(dim=-1).max().item()))
        if k_max > 0:
            _, sel = unmask_score.topk(k=k_max, dim=-1)
            valid = mask_indices.gather(1, sel)
            rank = torch.arange(k_max, device=xt.device).unsqueeze(0)
            valid_rank = rank < mask_indices.sum(dim=-1, keepdim=True).clamp(max=k_max)
            update_mask = torch.zeros_like(mask_indices)
            update_mask.scatter_(1, sel, valid & valid_rank)
            new_tokens = logits_noisy.argmax(dim=-1)
            xt_new = torch.where(update_mask, new_tokens, xt_new)

        remasked_counts = torch.zeros(B, device=xt.device, dtype=torch.long)
        if stride > 0 and step % stride == 0 and remask_budget > 0:
            stats["correction_steps"] += 1
            # Score xt (before this step's demasking) per Algorithm 1: P_error = g(x_t).
            # Candidate positions are those already clean in xt; they are also clean in
            # xt_new (demasking only reveals new tokens, never masks existing ones).
            clean = (xt != mask_id) & (~prompt_mask)
            if clean.sum() > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    _, hidden = model.forward_with_hidden(xt)
                error_prob = 1.0 - torch.sigmoid(model.adapter(hidden))
                clean_error_prob = error_prob[clean]
                if clean_error_prob.numel() > 0:
                    stats["score_count"] += float(clean_error_prob.numel())
                    stats["error_prob_sum"] += float(clean_error_prob.sum().item())
                    max_error = error_prob.masked_fill(~clean, float("-inf")).max(dim=-1).values
                    valid_max_error = max_error[torch.isfinite(max_error)]
                    if valid_max_error.numel() > 0:
                        stats["error_prob_max_sum"] += float(valid_max_error.sum().item())
                        stats["error_prob_max_count"] += float(valid_max_error.numel())
                error_score = error_prob.masked_fill(~clean, float("-inf"))
                k_err = min(remask_budget, int(clean.sum(dim=-1).max().item()))
                if k_err > 0:
                    _, candidate_idx = error_score.topk(k=k_err, dim=-1)
                    for b in range(B):
                        for idx in candidate_idx[b].tolist():
                            if not clean[b, idx]:
                                continue
                            if error_prob[b, idx].item() <= tau:
                                continue
                            if idx in hblock[b]:
                                continue
                            xt_new[b, idx] = mask_id
                            remasked_counts[b] += 1
                            any_remask[b] = True
                            hblock[b].append(idx)
                            if len(hblock[b]) > block_size:
                                hblock[b].pop(0)
                stats["total_remasked_tokens"] += float(remasked_counts.sum().item())

        t_new = max(t - delta_t, 0.0)
        # Approximate per-sequence rollback (t += |M_remask|/L) with the batch mean,
        # matching the paper's intent more closely than the previous max().
        rollback = remasked_counts.float().mean().item() / max(answer_len, 1)
        t = min(t_new + rollback, 1.0)
        xt = xt_new
        step += 1

        if track:
            track_list.append(xt.clone().detach().cpu())

    stats["sequences_with_any_remask"] = float(any_remask.sum().item())

    if track:
        if return_stats:
            return xt, torch.stack(track_list, dim=0), stats
        return xt, torch.stack(track_list, dim=0)
    if return_stats:
        return xt, stats
    return xt


def backplay_evaluate_ddp_sudoku(model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None):
    """
    Drop-in replacement for evaluate_ddp_sudoku that uses BackPlay sampling.
    """
    from eval.sudoku_eval import verify_sudoku

    val_dir = cfg.validation.val_dir
    mask_id = cfg.data.mask_id
    seq_len = cfg.model.max_position
    n4 = seq_len // 2
    n = round(n4 ** 0.25)

    test_mdm_path = os.path.join(val_dir, f"test_mdm_n{n}.npy")
    if not os.path.exists(test_mdm_path):
        test_mdm_path = os.path.join(val_dir, "test_mdm.npy")
    if not os.path.exists(test_mdm_path):
        raise FileNotFoundError(f"No test_mdm_n{n}.npy or test_mdm.npy in {val_dir}")

    X = np.load(test_mdm_path).copy()
    Y = np.load(test_mdm_path)
    X[:, n4:] = mask_id

    N = len(X)
    ratio = cfg.validation.ratio
    N_val = int(N * ratio)
    X, Y = X[:N_val], Y[:N_val]

    per_rank = math.ceil(N_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, N_val)
    batch_size = cfg.validation.get("batch_size", 64)
    num_batches = math.ceil((end - start) / batch_size)

    local_correct, local_total = 0, 0
    local_stats = {
        "num_steps": 0.0,
        "correction_steps": 0.0,
        "sequences_with_any_remask": 0.0,
        "total_remasked_tokens": 0.0,
        "score_count": 0.0,
        "error_prob_sum": 0.0,
        "error_prob_max_sum": 0.0,
        "error_prob_max_count": 0.0,
    }
    with torch.no_grad():
        for j in tqdm(range(num_batches), desc="BackPlay Eval", disable=(rank != 0)):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X = torch.from_numpy(X[s:e]).long().to(device)
            batch_Y = torch.from_numpy(Y[s:e]).long().to(device)
            prompt_mask = torch.zeros_like(batch_X, dtype=torch.bool)
            prompt_mask[:, :n4] = True

            pred, batch_stats = backplay_sampling(
                model,
                batch_X,
                mask_id,
                sampling,
                prompt_mask=prompt_mask,
                device=device,
                return_stats=True,
            )
            matches = verify_sudoku(pred, batch_Y, n)
            local_correct += matches.sum().item()
            local_total += batch_Y.shape[0]
            for key in local_stats:
                local_stats[key] += batch_stats[key]

    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    stats_tensor = torch.tensor([
        local_stats["num_steps"],
        local_stats["correction_steps"],
        local_stats["sequences_with_any_remask"],
        local_stats["total_remasked_tokens"],
        local_stats["score_count"],
        local_stats["error_prob_sum"],
        local_stats["error_prob_max_sum"],
        local_stats["error_prob_max_count"],
    ], dtype=torch.float64, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    g_correct, g_total = tensor.tolist()
    denom = max(g_total, 1)
    score_count = max(float(stats_tensor[4].item()), 1.0)
    max_error_count = max(float(stats_tensor[7].item()), 1.0)
    stats = {
        "acc": g_correct / denom,
        "avg_steps_per_batch": float(stats_tensor[0].item()) / max(num_batches * world_size, 1),
        "avg_correction_steps_per_batch": float(stats_tensor[1].item()) / max(num_batches * world_size, 1),
        "remask_seq_frac": float(stats_tensor[2].item()) / denom,
        "avg_remasked_tokens_per_seq": float(stats_tensor[3].item()) / denom,
        "avg_error_prob": float(stats_tensor[5].item()) / score_count,
        "avg_max_error_prob_per_correction": float(stats_tensor[6].item()) / max_error_count,
    }
    return stats
