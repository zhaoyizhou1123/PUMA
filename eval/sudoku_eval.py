import torch
import os
import numpy as np
import math
import torch.distributed as dist
from sampling import mdm_sampling
from tqdm import tqdm

# sudoku eval helper

def evaluate_ddp_sudoku(model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None):
    val_dir = cfg.validation.val_dir
    mask_id = cfg.data.mask_id
    # track = cfg.validation.get("track", False)

    # infer n from model config max_position (= 2 * n^4)
    seq_len = cfg.model.max_position
    n4 = seq_len // 2
    n = round(n4 ** 0.25)

    # get the test Sudoku puzzle and answers
    # prefer n-specific naming, fall back to legacy test_mdm.npy
    test_mdm_path = os.path.join(val_dir, f"test_mdm_n{n}.npy")
    if not os.path.exists(test_mdm_path):
        test_mdm_path = os.path.join(val_dir, "test_mdm.npy")
    if not os.path.exists(test_mdm_path):
        raise FileNotFoundError(f"No test_mdm_n{n}.npy or test_mdm.npy found in {val_dir}")

    X, Y = np.load(test_mdm_path), np.load(test_mdm_path)
    X = X.copy()

    X[:, n4:] = cfg.data.mask_id
    N = len(X)
    # for our initial runs, we split the validation set (time efficiency)
    ratio = cfg.validation.ratio
    N_val = int(N * ratio)
    X, Y = X[:N_val], Y[:N_val]

    # distribute test cases
    per_rank = math.ceil(N_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, N_val)

    batch_size = cfg.validation.get('batch_size', 64)
    num_batches = math.ceil((end - start) / batch_size)
    local_correct, local_total = 0, 0

    with torch.no_grad():
        for j in tqdm(range(num_batches), desc = "Evaluating"):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X = torch.from_numpy(X[s:e]).long().to(device)
            batch_Y = torch.from_numpy(Y[s:e]).long().to(device)
            # Create a prompt mask
            prompt_mask = torch.zeros_like(batch_X, dtype=torch.bool)
            prompt_mask[:, :n4] = True

            if not cfg.validation.track or j >= 1:
                pred = mdm_sampling(model, batch_X, mask_id, sampling, device, prompt_mask=prompt_mask)
            else: # Only track the first batch for visualization/debugging
                pred, track_xt = mdm_sampling(model, batch_X, mask_id, sampling, device, prompt_mask=prompt_mask, track=True)
                track_xt = track_xt.cpu().numpy()
                np.save(os.path.join(logdir, f"step{step}_rank{rank}.npy"), track_xt)

            matches = verify_sudoku(pred, batch_Y, n)
            local_correct += matches.sum().item()
            local_total += batch_Y.shape[0]

    # accumulate succcess rates
    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()

    return global_correct / global_total

def verify_sudoku(pred: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    """
    pred: [B, 2*n^4] where pred[:, :n^4] are clues/condition, pred[:, n^4:] is predicted solution
    target: [B, 2*n^4] where target[:, :n^4] are clues/condition, target[:, n^4:] is ground-truth solution
    returns: [B] bool
    """
    n4 = n ** 4
    cond = pred[:, :n4]
    sol  = pred[:, n4:]

    clue_ok = ((cond == 0) | (sol == cond)).all(dim=1)   # [B]
    sudoku_ok = sudoku_check(sol, n)                      # [B]

    return clue_ok & sudoku_ok


def sudoku_check(pred: torch.Tensor, n: int) -> torch.Tensor:
    """
    Check if the predicted Sudoku solution is valid.
    pred: [B, n^4], returns [B] bool
    """
    n2 = n ** 2
    B, _ = pred.shape
    x = pred.view(B, n2, n2)

    # Must be integers in {1,...,n²} (no zeros allowed in a completed Sudoku)
    in_range = (x >= 1) & (x <= n2)

    # Helper: check each length-n² group is a permutation of 1..n²
    ref = torch.arange(1, n2 + 1, device=pred.device, dtype=pred.dtype).view(1, 1, n2)

    def groups_ok(groups: torch.Tensor) -> torch.Tensor:
        # groups: [B, G, n²]
        sorted_groups, _ = torch.sort(groups, dim=-1)
        return (sorted_groups == ref).all(dim=-1)  # [B, G] bool

    # Rows: [B, n², n²]
    rows_ok = groups_ok(x)

    # Cols: [B, n², n²]
    cols_ok = groups_ok(x.transpose(1, 2))

    # n×n blocks: reshape into n² blocks of n²
    blocks = x.view(B, n, n, n, n).permute(0, 1, 3, 2, 4).contiguous().view(B, n2, n2)
    blocks_ok = groups_ok(blocks)

    # All constraints must hold + all entries in range
    return in_range.all(dim=(1, 2)) & rows_ok.all(dim=1) & cols_ok.all(dim=1) & blocks_ok.all(dim=1)
