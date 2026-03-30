import torch
import os
import numpy as np
import math
import torch.distributed as dist
from sampling import mdm_sampling
from tqdm import tqdm

# sudoku eval helper

def evaluate_ddp_maze(model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None, metric_name=""):
    val_dir = cfg.validation.val_dir
    mask_id = cfg.data.mask_id

    test_inputs = os.path.join(val_dir, "test_labels.npy")
    test_answers = os.path.join(val_dir, "test_labels.npy")
    prompt_mask_path = os.path.join(val_dir, "test_prompt_mask.npy")

    X, Y  = np.load(test_inputs), np.load(test_answers)
    prompt_masks = np.load(prompt_mask_path)
    X = X.copy()
    X[~prompt_masks] = cfg.data.mask_id
    N = len(X)
    # for our initial runs, we split the validation set (time efficiency)
    ratio = cfg.validation.ratio
    N_val = int(N * ratio)
    print(f"Total test cases: {N}, using {N_val} for evaluation (ratio={ratio})")
    X, Y = X[:N_val], Y[:N_val]
    prompt_masks = prompt_masks[:N_val]

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
            batch_X = torch.from_numpy(X[s:e]).long().to(device) # (B, 162)
            batch_Y = torch.from_numpy(Y[s:e]).long().to(device)
            # Create a prompt mask
            # prompt_mask = torch.zeros_like(batch_X, dtype=torch.bool)
            # prompt_mask[:, :81] = True
            prompt_mask = torch.from_numpy(prompt_masks[s:e]).to(device)

            do_track = cfg.validation.track and j == 0
            result = mdm_sampling(model, batch_X, mask_id, sampling, device, prompt_mask=prompt_mask, track=do_track)
            if do_track:
                pred, extra_info = result
                if logdir is not None:
                    os.makedirs(logdir, exist_ok=True)
                    with open(os.path.join(logdir, metric_name + f"step{step}_rank{rank}.pkl"), "wb") as f:
                        pickle.dump(extra_info, f)
            else:
                pred = result

            matches = (pred == batch_Y).all(dim=1) # verify exact match for the whole solution
            local_correct += matches.sum().item()
            local_total += batch_Y.shape[0]

    # accumulate succcess rates
    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()

    return global_correct / global_total

def verify_sudoku(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred: [B, 162] where pred[:, :81] are clues/condition, pred[:, 81:] is predicted solution
    target: [B, 162] where target[:, :81] are clues/condition, target[:, 81:] is ground-truth solution
    returns: [B] bool
    """
    cond = pred[:, :81]
    sol  = pred[:, 81:]

    clue_ok = ((cond == 0) | (sol == cond)).any(dim=1)   # [B]
    sudoku_ok = sudoku_check(sol)                        # [B]

    return clue_ok & sudoku_ok 
    

    

def sudoku_check(pred: torch.Tensor) -> torch.Tensor:
    """
    Check if the predicted Sudoku solution is valid.
    pred: [B, 81], returns [B] bool
    """
    B, _ = pred.shape
    x = pred.view(B, 9, 9)

    # Must be integers in {1,...,9} (no zeros allowed in a completed Sudoku)
    in_range = (x >= 1) & (x <= 9)

    # Helper: check each length-9 group is a permutation of 1..9
    ref = torch.arange(1, 10, device=pred.device, dtype=pred.dtype).view(1, 1, 9)

    def groups_ok(groups: torch.Tensor) -> torch.Tensor:
        # groups: [B, G, 9]
        sorted_groups, _ = torch.sort(groups, dim=-1)
        return (sorted_groups == ref).all(dim=-1)  # [B, G] bool

    # Rows: [B, 9, 9]
    rows_ok = groups_ok(x)

    # Cols: [B, 9, 9]
    cols_ok = groups_ok(x.transpose(1, 2))

    # 3x3 blocks: reshape into 9 blocks of 9
    blocks = x.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).contiguous().view(B, 9, 9)
    blocks_ok = groups_ok(blocks)

    # All constraints must hold + all entries in range
    return in_range.all(dim=(1, 2)) & rows_ok.all(dim=1) & cols_ok.all(dim=1) & blocks_ok.all(dim=1)
