"""
ReMDM inference — remasking discrete diffusion sampling.
Reference: arXiv:2503.00307v4 (Kuleshov group)

Variants
--------
cap:     sigma = min(eta, (1 - alpha_s) / alpha_t)
rescale: sigma = eta * min(1, (1 - alpha_s) / alpha_t)
conf:    per-token sigma^(l) = softmax(conf) * sigma_max, where conf tracks
         negative decode-time confidence for currently decoded tokens.

Adaptation note
---------------
This implementation now follows the paper/upstream sampler structure:
  1. choose a reverse-time grid with num_steps,
  2. compute alpha_t, alpha_s from an explicit noise schedule,
  3. sample every non-prompt token from the ReMDM approximate posterior.

Task-level adaptation: Sudoku prompts are pinned via prompt_mask.

Important: MDLM vs ReMDM (upstream diffusion._ddpm_caching_update)
--------------------------------------------------------------------
MDLM keeps already-unmasked tokens: xs = x on non-mask positions.
ReMDM merges q distributions then _sample_categorical on *all* positions, so
decoded tokens are redrawn from p_x0*(1-sigma) each step (unless frozen).
PUMA evaluate.py uses mdm_sampling — frozen unmasking — so its accuracy is
not comparable to textbook ReMDM without freeze_unmasked=True.

Upstream alignment (kuleshov-group/remdm diffusion._ddpm_caching_update)
-------------------------------------------------------------------------
* time_grid: "mdlm" — t_i = linspace(1, time_eps, num_steps+1), same as their
  _sample loop; alpha_t = 1 - t, alpha_s = 1 - t_next (move_chance = t).
* time_grid: "uniform" — legacy grid t = step/num_steps (backward steps).
* nucleus_p < 1 — optional top-p filter on p_x0 before the ReMDM update.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    """
    Sample categorical variables with fp64 Gumbel noise, matching the upstream
    repository's implementation choice.
    """
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _alpha_from_time(
    t: torch.Tensor,
    schedule: str = "loglinear",
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Map reverse-process time t in [0, 1] to alpha_t.

    For the upstream loglinear schedule, alpha_t = exp(-sigma_t) simplifies to
    1 - (1 - eps) * t.
    """
    if schedule == "loglinear":
        return 1.0 - (1.0 - eps) * t
    if schedule == "linear":
        return 1.0 - t
    raise ValueError(f"Unknown alpha schedule: {schedule!r}")


def _sigma_max(alpha_t: torch.Tensor, alpha_s: torch.Tensor) -> torch.Tensor:
    sigma_max = torch.ones_like(alpha_t)
    active = alpha_t > 0
    sigma_max[active] = torch.minimum(
        torch.ones_like(alpha_t[active]),
        (1.0 - alpha_s[active]) / alpha_t[active],
    )
    return sigma_max


def _apply_nucleus(p_x0: torch.Tensor, nucleus_p: float) -> torch.Tensor:
    """Top-p on p_x0 (same pattern as upstream remdm diffusion._ddpm_caching_update)."""
    if nucleus_p >= 1.0:
        return p_x0
    sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs <= nucleus_p
    top_p_mask[..., 0] = True
    nucleus_probs = sorted_probs * top_p_mask
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)


@torch.no_grad()
def remdm_sampling(
    model,
    xt: torch.Tensor,                  # [B, L] — answer positions = mask_id
    mask_id: int,
    sampling_cfg,
    prompt_mask: torch.Tensor = None,  # [B, L] bool; True = prompt (pinned)
    device: torch.device = None,
) -> torch.Tensor:
    """
    ReMDM sampler.  Returns [B, L] fully decoded sequence.

    sampling_cfg fields
    -------------------
    variant       : str   — "cap" | "rescale" | "conf"
    eta           : float — remasking intensity in [0, 1]
    unmasking_num : int   — legacy fallback only; prefer num_steps in config
    temperature   : float — optional posterior temperature; <=0 means 1.0
    time_grid     : str   — "mdlm" (match upstream grid + alpha=1-t) | "uniform" (legacy)
    time_eps      : float — end of linspace for time_grid=mdlm (default 1e-5)
    nucleus_p     : float — top-p on p_x0; 1.0 disables (default)
    freeze_unmasked : bool — if True, keep current token on non-mask cells (like
        MDLM / PUMA mdm_sampling). False = strict upstream ReMDM (resample all).
    """
    variant       = getattr(sampling_cfg, "variant", "cap")
    eta           = float(getattr(sampling_cfg, "eta", 0.0))
    unmasking_num = int(sampling_cfg.unmasking_num)
    temperature   = float(sampling_cfg.temperature)
    time_grid     = getattr(sampling_cfg, "time_grid", "uniform")
    time_eps      = float(getattr(sampling_cfg, "time_eps", 1e-5))
    nucleus_p     = float(getattr(sampling_cfg, "nucleus_p", 1.0))
    alpha_schedule = getattr(sampling_cfg, "alpha_schedule", "loglinear")
    alpha_eps = float(getattr(sampling_cfg, "alpha_eps", 1e-3))
    conf_base_variant = getattr(sampling_cfg, "conf_base_variant", "repo")
    freeze_unmasked = bool(getattr(sampling_cfg, "freeze_unmasked", False))

    if prompt_mask is None:
        prompt_mask = torch.zeros_like(xt, dtype=torch.bool)

    B, L = xt.shape
    xt = xt.clone()
    valid_mask = ~prompt_mask
    L_eff = valid_mask.sum(dim=1).clamp(min=1)

    # Upstream ReMDM-conf tracks negative decode-time confidence and then
    # normalizes over currently decoded tokens with a softmax.
    if variant == "conf":
        conf_scores = torch.full((B, L), float("-inf"), device=xt.device)

    max_eff_len = int(L_eff.max().item())
    num_steps = int(getattr(
        sampling_cfg,
        "num_steps",
        math.ceil(max_eff_len / max(unmasking_num, 1)),
    ))
    posterior_temp = 1.0 if temperature <= 0.0 else temperature

    if time_grid == "mdlm":
        ts = torch.linspace(
            1.0,
            time_eps,
            num_steps + 1,
            device=xt.device,
            dtype=torch.float32,
        )
    elif time_grid != "uniform":
        raise ValueError(
            f"Unknown time_grid: {time_grid!r}. Expected 'uniform' or 'mdlm'."
        )

    if time_grid == "mdlm":
        step_range = range(num_steps)
    else:
        step_range = range(num_steps, 0, -1)

    for step in step_range:
        mask_indices = (xt == mask_id) & valid_mask
        if mask_indices.sum() == 0:
            break

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            logits = model(xt)
        logits = logits.float()

        p_x0 = F.softmax(logits / posterior_temp, dim=-1)
        p_x0 = _apply_nucleus(p_x0, nucleus_p)

        if time_grid == "mdlm":
            i = step
            t = torch.full((B,), ts[i].item(), device=xt.device, dtype=logits.dtype)
            s = torch.full((B,), ts[i + 1].item(), device=xt.device, dtype=logits.dtype)
            # Upstream: move_chance = t, alpha = 1 - move_chance
            alpha_t = 1.0 - t
            alpha_s = 1.0 - s
        else:
            t = torch.full((B,), step / num_steps, device=xt.device, dtype=logits.dtype)
            s = torch.full((B,), (step - 1) / num_steps, device=xt.device, dtype=logits.dtype)
            alpha_t = _alpha_from_time(t, schedule=alpha_schedule, eps=alpha_eps)
            alpha_s = _alpha_from_time(s, schedule=alpha_schedule, eps=alpha_eps)

        sigma_max = _sigma_max(alpha_t, alpha_s)

        if variant == "cap":
            sigma = torch.minimum(torch.full_like(alpha_t, eta), sigma_max)
            sigma = sigma[:, None]
        elif variant == "rescale":
            sigma = (eta * sigma_max)[:, None]
        elif variant == "conf":
            decoded_mask = (~mask_indices) & valid_mask
            eta_weights = torch.zeros_like(conf_scores)
            if decoded_mask.any():
                safe_conf = conf_scores.masked_fill(~decoded_mask, float("-inf"))
                has_decoded = decoded_mask.any(dim=-1)
                eta_weights[has_decoded] = safe_conf[has_decoded].softmax(dim=-1)
            if conf_base_variant == "repo":
                sigma_base = sigma_max
            elif conf_base_variant == "cap":
                sigma_base = torch.minimum(torch.full_like(alpha_t, eta), sigma_max)
            elif conf_base_variant == "rescale":
                sigma_base = eta * sigma_max
            else:
                raise ValueError(
                    f"Unknown conf_base_variant: {conf_base_variant!r}. "
                    "Expected: repo / cap / rescale"
                )
            sigma = eta_weights * sigma_base[:, None]
        else:
            raise ValueError(
                f"Unknown ReMDM variant: {variant!r}.  Expected: cap / rescale / conf"
            )

        sigma_tokens = sigma if sigma.ndim == 2 else sigma.expand(B, L)
        sigma_tokens = torch.where(valid_mask, sigma_tokens, torch.zeros_like(sigma_tokens))

        alpha_t_exp = alpha_t[:, None, None]
        alpha_s_exp = alpha_s[:, None, None]
        sigma_exp = sigma_tokens[:, :, None]

        q_xs = p_x0 * (1.0 - sigma_exp)
        q_xs[..., mask_id] = sigma_tokens

        denom = (1.0 - alpha_t_exp).clamp(min=1e-6)
        q_xs_masked = p_x0 * ((alpha_s_exp - (1.0 - sigma_exp) * alpha_t_exp) / denom)
        q_xs_masked[..., mask_id] = (
            (1.0 - alpha_s[:, None]) - sigma_tokens * alpha_t[:, None]
        ) / denom.squeeze(-1)

        q_xs = torch.where(mask_indices.unsqueeze(-1), q_xs_masked, q_xs)
        q_xs = torch.where(valid_mask.unsqueeze(-1), q_xs, F.one_hot(xt, num_classes=logits.shape[-1]).float())
        q_xs = q_xs.clamp(min=0.0)
        q_xs = q_xs / q_xs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        xs = _sample_categorical(q_xs)
        if freeze_unmasked:
            keep = (~mask_indices) & valid_mask
            xs = torch.where(keep, xt, xs)

        if variant == "conf":
            unmask_mask = mask_indices & (xs != mask_id)
            batch_indices = torch.arange(B, device=xt.device)[:, None]
            feature_indices = torch.arange(L, device=xt.device)[None, :]
            conf_values = -p_x0[batch_indices, feature_indices, xs]
            conf_scores = torch.where(unmask_mask, conf_values, conf_scores)
            remask_mask = ((xt != mask_id) & valid_mask) & (xs == mask_id)
            conf_scores = torch.where(
                remask_mask,
                torch.full_like(conf_scores, float("-inf")),
                conf_scores,
            )

        xt = torch.where(valid_mask, xs, xt)

    return xt


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def remdm_evaluate_ddp_sudoku(
    model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None
):
    """
    Drop-in replacement for evaluate_ddp_sudoku that uses remdm_sampling.
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
        for j in tqdm(range(num_batches), desc="ReMDM Eval", disable=(rank != 0)):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X     = torch.from_numpy(X[s:e]).long().to(device)
            batch_Y     = torch.from_numpy(Y[s:e]).long().to(device)
            prompt_mask = torch.zeros_like(batch_X, dtype=torch.bool)
            prompt_mask[:, :n4] = True

            pred    = remdm_sampling(
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
