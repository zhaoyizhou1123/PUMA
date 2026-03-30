import math, os, time, json, random, sys, datetime
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torch.distributed as dist
import argparse
from copy import deepcopy
from tqdm import tqdm
from model.transformer import MDMTransformer, MDMConfig
from data import setup_data_bundle
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from typing import Optional, List, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig, ListConfig, open_dict
from model.ema import ExponentialMovingAverage, save_ema_snapshot, save_model_snapshot
from progressive import PhasedMaskingEdit, PhasedMaskingEditV2, PhasedMasking, mdm_edit_loss_fn, mdm_loss_fn, PhasedMaskingEditV3, mdm_edit_loss_fn_v5
from eval.sudoku_eval import evaluate_ddp_sudoku
from eval.gsm8k_eval import evaluate_ddp_gsm8k
from data.sudoku_utils import resolve_sudoku_grid_size

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cfg", type=str)
#     return parser.parse_args()


def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank

def compute_scaled_learning_rate(base_lr: float, batch_size: int, scaling_mode: str, base_batch_size: int = 128) -> float:
    """
    Compute scaled learning rate based on batch size.

    Args:
        base_lr: Base learning rate (for batch_size=base_batch_size)
        batch_size: Current batch size
        scaling_mode: One of 'linear', 'sqrt', or 'constant'
        base_batch_size: Reference batch size (default: 128)

    Returns:
        Scaled learning rate
    """
    if scaling_mode == "constant":
        return base_lr
    elif scaling_mode == "sqrt":
        scale_factor = math.sqrt(batch_size / base_batch_size)
    elif scaling_mode == "linear":
        scale_factor = batch_size / base_batch_size
    else:
        raise ValueError(f"Invalid lr_scaling_mode: {scaling_mode}. Must be 'linear', 'sqrt', or 'constant'")

    scaled_lr = base_lr * scale_factor
    return scaled_lr


def evaluate_ddp_dict(model, cfg, device, rank, world_size, step=0, logdir=None):
    sampling = cfg.validation.sampling
    if cfg.training.strategy == "arm":
        return {"arm": evaluate_ddp(model, cfg, device, rank, world_size, sampling, step=step, logdir=logdir)}
    base_sampling = sampling
    out = {}

    # Determine if strategy uses editing during training
    strategy_uses_edit = cfg.training.strategy in ["proseco", "edit", "progressive_edit"]

    # Always get the full list of edit_freq and edit_step from config for metric naming
    edit_freq_list = list(base_sampling.edit_freq) if hasattr(base_sampling, "edit_freq") else [None]
    edit_step_list = list(base_sampling.edit_step) if hasattr(base_sampling, "edit_step") else [None]

    for confidence in list(base_sampling.confidence):
        for unmasking_num in list(base_sampling.unmasking_num):
            # For strategies without editing, evaluate once and reuse the result
            if not strategy_uses_edit:
                # Evaluate with no editing (edit_freq=-1)
                sampling_no_edit = deepcopy(base_sampling)
                with open_dict(sampling_no_edit):
                    sampling_no_edit.confidence = confidence
                    sampling_no_edit.unmasking_num = unmasking_num
                    sampling_no_edit.edit_freq = -1
                    if hasattr(base_sampling, "edit_step") and len(list(base_sampling.edit_step)) > 0:
                        sampling_no_edit.edit_step = list(base_sampling.edit_step)[0]
                metric_name_no_edit = f"{confidence}_unmasking_{unmasking_num}_editfreq_-1_editstep_{sampling_no_edit.edit_step if hasattr(sampling_no_edit, 'edit_step') else 'none'}"
                result_no_edit = evaluate_ddp(model, cfg, device, rank, world_size, sampling_no_edit, step=step, logdir=logdir)

                # Fill all edit_freq variants with the same result for wandb consistency
                for edit_freq in edit_freq_list:
                    for edit_step in edit_step_list:
                        metric_name = f"{confidence}_unmasking_{unmasking_num}"
                        if edit_freq is not None:
                            metric_name += f"_editfreq_{edit_freq}"
                        if edit_step is not None:
                            metric_name += f"_editstep_{edit_step}"
                        out[metric_name] = result_no_edit
            else:
                # For edit-based strategies, evaluate each edit_freq separately
                for edit_freq in edit_freq_list:
                    for edit_step in edit_step_list:
                        sampling = deepcopy(base_sampling)
                        sampling.confidence = confidence
                        sampling.unmasking_num = unmasking_num
                        metric_name = f"{confidence}_unmasking_{unmasking_num}"
                        if edit_freq is not None:
                            sampling.edit_freq = edit_freq
                            metric_name += f"_editfreq_{edit_freq}"
                        if edit_step is not None:
                            sampling.edit_step = edit_step
                            metric_name += f"_editstep_{edit_step}"
                        out[metric_name] = evaluate_ddp(model, cfg, device, rank, world_size, sampling, step=step, logdir=logdir)
    return out

def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.norm(p=2).item()
    return total ** 0.5

def evaluate_ddp(model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None):
    if cfg.data.dataset == "sudoku":
        return evaluate_ddp_sudoku(model, cfg, device, rank, world_size, sampling, step=step, logdir=logdir)
    elif cfg.data.dataset == "tinygsm":
        return evaluate_ddp_gsm8k(model, cfg, device, rank, world_size, sampling)
    elif cfg.data.dataset == "maze":
        from eval.maze_eval import evaluate_ddp_maze
        return evaluate_ddp_maze(model, cfg, device, rank, world_size, sampling, step=step, logdir=logdir)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")

# mdm loss implementation, but computes all indices
def mdm_loss_edit(model, input_ids, mask_id: int, prompt_mask: Optional[torch.Tensor] = None, arm_init: bool = False):
    # sample integer uniformly for each batch from [1,L]
    # prompt_mask (boolean mask): 1 for prompt
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device
    B, L = input_ids.shape
    L_eff = L - prompt_mask.sum(dim=1 , keepdim=True)
    valid = ~prompt_mask
    # uniformly sample the number of positions to mask
    num_mask = torch.floor(torch.rand(B, 1, device=device) * L_eff.clamp(min=1)).long() + 1

    # mask correspondent number of tokens for each batch, 0.0 for the prompt indices
    scores = torch.rand((B, L), device=device).masked_fill(prompt_mask, float('inf')).argsort(dim=1)
    order = scores.argsort(dim=1)
    mask_indices = (order < num_mask)
    masked_input = torch.where(mask_indices, mask_id, input_ids)
    logits = model(masked_input)

    # calculate (reweighted) loss
    num_mask = num_mask.float().expand_as(mask_indices)

    if arm_init:
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        valid = valid[:, 1:]
    return F.cross_entropy(logits[valid], input_ids[valid], reduction="mean")

def mdm_loss(model, input_ids, mask_id: int, prompt_mask: Optional[torch.Tensor] = None, arm_init: bool = False):
    # sample integer uniformly for each batch from [1,L]
    # prompt_mask (boolean mask): 1 for prompt
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device
    B, L = input_ids.shape
    L_eff = L - prompt_mask.sum(dim=1 , keepdim=True)
    # uniformly sample the number of positions to mask
    num_mask = torch.floor(torch.rand(B, 1, device=device) * L_eff.clamp(min=1)).long() + 1

    # mask correspondent number of tokens for each batch, 0.0 for the prompt indices
    scores = torch.rand((B, L), device=device).masked_fill(prompt_mask, float('inf')).argsort(dim=1)
    order = scores.argsort(dim=1)
    mask_indices = (order < num_mask)
    masked_input = torch.where(mask_indices, mask_id, input_ids)
    logits = model(masked_input)

    # calculate (reweighted) loss
    num_mask = num_mask.float().expand_as(mask_indices)

    if arm_init:
        ce = F.cross_entropy(logits[:, :-1, :][mask_indices[:, 1:]], input_ids[:, 1:][mask_indices[:, 1:]], reduction="none")
    else:
        ce = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction="none")
    loss = ce / num_mask[mask_indices]
    return loss.sum() / B

def arm_loss(
    model,
    input_ids: torch.Tensor,                    # (B, L)
    eos_id: int,
    prompt_mask: Optional[torch.Tensor] = None, # True = prompt token
):
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    logits = model(input_ids)          # (B, L, V)
    targets = input_ids[:, 1:]         # (B, L-1)
    pred_logits = logits[:, :-1, :]    # (B, L-1, V)

    valid = ~prompt_mask[:, 1:]        # (B, L-1)

    if eos_id is not None:
        is_eos = (targets == eos_id)               # (B, L-1)
    else:
        is_eos = torch.zeros_like(targets, dtype=torch.bool)
    any_eos = is_eos.any(dim=1)                # (B,)
    first_eos = is_eos.float().argmax(dim=1)   # (B,) 0-based in targets
    first_eos = torch.where(
        any_eos,
        first_eos,
        torch.full_like(first_eos, targets.shape[1] - 1),
    )

    t = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)  # (1, L-1)
    valid = valid & (t <= first_eos.unsqueeze(1))

    if valid.sum().item() == 0:
        return pred_logits.sum() * 0.0
    return F.cross_entropy(pred_logits[valid], targets[valid], reduction="mean")

# validation loss helper
def val_loss_ddp(model, val_loader, mask_id: int, device, rank: int, world_size: int, strategy: str, eos_id: int, arm_init: bool = False):
    model.eval()
    if world_size > 1 and dist.is_initialized() and not isinstance(val_loader.sampler, DistributedSampler):
        sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size or 16,
        sampler=sampler,
        num_workers=0,  # must be 0: CUDA already initialized in main process, forked workers can't re-init
        pin_memory=False,
        drop_last=False,
        )
    else:
        sampler = None

    local_sum = 0.0
    local_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc = "Validating", disable = (rank != 0)):
            x0 = batch["labels"].to(device)
            pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
            
            # to enable flashattention, we do autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                if strategy == "arm":
                    loss = arm_loss(model, x0, eos_id=eos_id, prompt_mask=pm)
                elif strategy in ["progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v5", "edit_v6", "edit_v7"]:
                    loss = mdm_loss_edit(model, x0, mask_id, prompt_mask = pm, arm_init=arm_init)
                elif strategy in ["standard", "progressive"]:
                    loss = mdm_loss(model, x0, mask_id, prompt_mask = pm, arm_init=arm_init)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
            B = x0.shape[0]
            local_sum += float(loss.item() * B)
            local_count += B
    
    tensor = torch.tensor([local_sum, local_count], dtype=torch.float, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_sum, global_count = tensor.tolist()

    return global_sum / max(int(global_count), 1)

def parse_k_schedule_increasing(k_schedule) -> List[Tuple[int, int]]:
    """
    Expects k_schedule as an *already increasing* list of [K, step] pairs.
    Validates:
      - each entry is [K, step]
      - steps are strictly increasing
      - (optionally) first step is 0
    Returns list of (K, step) in the same order.
    """
    if k_schedule is None:
        return []

    sched = []
    prev_step = None
    for item in list(k_schedule):
        if not isinstance(item, (list, tuple, ListConfig)) or len(item) != 2:
            raise ValueError(f"k_schedule entries must be [K, step], got {item}")
        K, step = int(item[0]), int(item[1])
        if K <= 0:
            raise ValueError(f"Invalid K in k_schedule: {K}")
        if step < 0:
            raise ValueError(f"Invalid step in k_schedule: {step}")

        if prev_step is not None and step <= prev_step:
            raise ValueError(
                f"k_schedule must have strictly increasing steps, but got step {step} after {prev_step}. "
                f"Full schedule: {list(k_schedule)}"
            )

        sched.append((K, step))
        prev_step = step

    return sched


@hydra.main(version_base=None, config_path="../yaml_files/maze", config_name="maze")
def main(cfg: DictConfig):
    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start training!")
        print(f"Training with {world_size} GPUs")

    # Resolve grid_size -> vocab_size, max_position, mask_id for sudoku configs
    cfg = resolve_sudoku_grid_size(cfg)

    # Manually compute per-GPU batch size and set it in the config
    bs = cfg.training.batch_size
    assert bs % world_size == 0, f"Batch size {bs} must be divisible by world size {world_size}"
    with open_dict(cfg):
        cfg.data.training.per_gpu_batch_size = bs // world_size
    
    base_seed = cfg.data.seed
    seed = base_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # ckpt dir
    datetime_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    ckpt_root = cfg.training.get("ckpt_root", "ckpts")
    ckpt_dir = f"{ckpt_root}/{cfg.wandb.project}/{cfg.wandb.name}_date{datetime_str}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    if cfg.validation.get("track", False):
        track_dir = f"track/date={datetime_str}"
        os.makedirs(track_dir, exist_ok=True)
        if is_main:
            print(f"Tracking files will be saved to: {track_dir}")
    else:
        track_dir = None

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize the model
    model_cfg_dict = cfg.model
    model_config = MDMConfig(**model_cfg_dict)
    model = MDMTransformer(model_config).to(device)

    # ARM initialization
    arm_init_path = model_cfg_dict.get("arm_init", "none")
    if arm_init_path != "none":
        model_config.predict_next_token = True
        if is_main:
            print(f"Initializing MDM from ARM checkpoint: {arm_init_path}")
        arm_ckpt = torch.load(arm_init_path, map_location="cpu")
        sd = arm_ckpt.get("model_state_dict", arm_ckpt)
        model.load_state_dict(sd, strict=True)


    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model is ready, parameters: {num_params/1e6:.2f}M")

    # model wrapping
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        if is_main:
            print(f"Model wrapping is done!")

    # data
    data_cfg = cfg.data
    train_cfg = cfg.training
    assert train_cfg.save_steps % train_cfg.eval_steps == 0, "save_steps must be divisible by eval_steps"
    val_cfg = cfg.validation
    data_bundle = setup_data_bundle(data_cfg)
    train_loader, val_loader = data_bundle.train_loader, data_bundle.val_loader    
    mask_id = data_cfg.mask_id
    eos_id = getattr(val_cfg.sampling, "eos_id", None)

    # training hyperparemeters
    # attach DDP sampler
    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            # batch_size=train_cfg.batch_size,
            batch_size=data_cfg.training.per_gpu_batch_size, # dataloader batch_size is micro batch size
            sampler=train_sampler,
            num_workers=0,  # must be 0 in DDP: CUDA already initialized, forked workers can't re-init
            pin_memory=False,
            drop_last=False
        )
    else:
        train_sampler = None

    # optimizer and scheduler
    # Compute scaled learning rate based on batch size
    lr_scaling_mode = getattr(train_cfg, "lr_scaling_mode", "constant")
    base_lr = train_cfg.learning_rate
    scaled_lr = compute_scaled_learning_rate(
        base_lr=base_lr,
        batch_size=train_cfg.batch_size,
        scaling_mode=lr_scaling_mode
    )

    if is_main:
        print(f"LR scaling mode: {lr_scaling_mode}")
        print(f"Base LR (batch_size=128): {base_lr}")
        print(f"Current batch size: {train_cfg.batch_size}")
        print(f"Scaled LR: {scaled_lr}")

    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=train_cfg.weight_decay)

    # Compute total training steps: use max_steps if provided, otherwise fall back to num_epochs
    num_training_steps = getattr(train_cfg, "max_steps", None)
    if num_training_steps is None:
        num_training_steps = train_cfg.num_epochs * len(train_loader)
    assert num_training_steps > 0, "training.max_steps must be positive"

    # Compute max_epochs needed to reach num_training_steps
    max_epochs = max(1, (num_training_steps + len(train_loader) - 1) // len(train_loader))

    if is_main:
        if getattr(train_cfg, "max_steps", None) is not None:
            print(f"Training for max_steps={num_training_steps} (~{max_epochs} epochs needed, num_epochs={train_cfg.num_epochs} ignored)")
        else:
            print(f"Training for {train_cfg.num_epochs} epochs ({num_training_steps} total steps)")

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps)
    if train_cfg.ema is not None:
        assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
        model_to_ema = model.module if isinstance(model, DDP) else model
        ema_params = [p for p in model_to_ema.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
        if is_main:
            print("EMA is enabled with decay:", train_cfg.ema)

    # Resume from checkpoint
    resume_path = cfg.training.get("resume", None)
    resume_global_step = 0
    if resume_path and resume_path != "none":
        if is_main:
            print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if train_cfg.ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.move_shadow_params_to_device(device)
        resume_global_step = ckpt.get("global_step", 0)
        if is_main:
            print(f"Resumed at global_step={resume_global_step}")

    strategy = train_cfg.strategy
    # k schedule for progressive_edit unmasking. If None use fixed K. If "linear", linearly increase the unmasking steps from 1 to K over the training steps.
    # If a list of integers, use the list as the k_steps. If an integer, use constant interval increase.
    if strategy in ["progressive", "progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v5", "edit_v6", "edit_v7"]: # puma, ours
        k_schedule = parse_k_schedule_increasing(getattr(train_cfg, "k_schedule", None))
        if len(k_schedule) == 0:
            k_schedule = [(train_cfg.K, 0)]
        
        current_k = k_schedule[0][0]

        if is_main:
            print("Using K Schedule:")
            for K, step in k_schedule:
                print(f"Step {step}: K={K}")

        # intialize the pool
        B = data_cfg.training.per_gpu_batch_size
        L = model_config.max_position
        def make_pool(K):
            B = data_cfg.training.per_gpu_batch_size
            L = model_config.max_position
            
            if strategy == "progressive":
                return PhasedMasking(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "progressive_edit":
                return PhasedMaskingEdit(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v2":
                return PhasedMaskingEditV2(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v3":
                return PhasedMaskingEditV3(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v3_2":
                from progressive import PhasedMaskingEditV3_2
                return PhasedMaskingEditV3_2(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v3_3":
                from progressive import PhasedMaskingEditV3_3
                return PhasedMaskingEditV3_3(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v4":
                from progressive import PhasedMaskingEditV4
                return PhasedMaskingEditV4(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v5":
                # still use v4 
                from progressive import PhasedMaskingEditV4
                return PhasedMaskingEditV4(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v6":
                from progressive import PhasedMaskingEditV6
                return PhasedMaskingEditV6(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            if strategy == "edit_v7":
                from progressive import PhasedMaskingEditV7
                return PhasedMaskingEditV7(
                    train_loader, B, mask_id, K, device, L,
                    mode=train_cfg.mode,
                    confidence_threshold=train_cfg.confidence_threshold,
                    eos_id=train_cfg.eos_id,
                )
            raise ValueError(f"Unknown strategy: {strategy}")
        pool = make_pool(current_k)
        next_k_idx = 1

        # fast-forward k_schedule to resume_global_step
        if resume_global_step > 0:
            for idx in range(1, len(k_schedule)):
                if resume_global_step >= k_schedule[idx][1]:
                    current_k = k_schedule[idx][0]
                    next_k_idx = idx + 1
                else:
                    break
            pool = make_pool(current_k)
            if is_main:
                print(f"[Resume] Restored K={current_k}, next_k_idx={next_k_idx}")


    # training loop
    steps_per_epoch = len(train_loader)
    skip_until_epoch = resume_global_step // steps_per_epoch
    resume_step_in_epoch = resume_global_step % steps_per_epoch
    global_step = resume_global_step

        
    # wandb initialize
    if cfg.wandb.wandb and is_main:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True))


    for epoch in range(max_epochs):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch < skip_until_epoch:
            continue

        step_offset = resume_step_in_epoch if epoch == skip_until_epoch else 0


        if strategy in ["progressive", "progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v5", "edit_v6", "edit_v7"]:
            pool.reset_loader_iter()
            iterable = range(steps_per_epoch)
        elif strategy == "standard" or strategy == "arm":
            iterable = train_loader

        if is_main:
            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}")
        else:
            pbar = iterable

        for step_in_epoch, itr in enumerate(pbar):
            # skip steps already completed before resume
            if step_in_epoch < step_offset:
                if strategy in ["progressive", "progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v5", "edit_v6", "edit_v7"]:
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                            xt = pool.current_batch()
                            logits = model(xt)
                            log_probs = F.log_softmax(logits, dim=-1)
                    pool.update_with_logits(log_probs)
                continue
            # Check if we've reached max_steps
            if global_step >= num_training_steps:
                break


            # update current K if using k schedule
            if strategy in ["progressive", "progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v5", "edit_v6", "edit_v7"] and next_k_idx < len(k_schedule) and global_step == k_schedule[next_k_idx][1]:
                current_k = k_schedule[next_k_idx][0]
                if is_main:
                    print(f"[K-SWITCH] Step {global_step}: K={current_k}")

                pool = make_pool(current_k)
                pool.reset_loader_iter()
                next_k_idx += 1

            # to enable flashattention, we do the autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                if strategy in ["progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v7"]:
                    xt = pool.current_batch()
                    # print("xt:", xt)
                    logits = model(xt)
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = mdm_edit_loss_fn(log_probs, pool.x0, pool.xt, mask_id, prompt_mask = pool.state['prompt_mask'], arm_init=model_config.predict_next_token)
                    # print(loss.shape)
                elif strategy in ["edit_v5", "edit_v6"]:
                    xt = pool.current_batch()
                    # print("xt:", xt)
                    logits = model(xt)
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = mdm_edit_loss_fn_v5(log_probs, pool.x0, pool.xt, mask_id, prompt_mask = pool.state['prompt_mask'], arm_init=model_config.predict_next_token)                   
                elif strategy == "progressive":
                    xt = pool.current_batch()
                    logits = model(xt)
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = mdm_loss_fn(log_probs, pool.x0, pool.xt, mask_id, prompt_mask = pool.state['prompt_mask'], arm_init=model_config.predict_next_token)
                elif strategy == "standard":
                    batch = itr
                    input_ids = batch["labels"].to(device)
                    prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                    loss = mdm_loss(model, input_ids, mask_id, prompt_mask = prompt_mask, arm_init=model_config.predict_next_token)
                elif strategy == "arm":
                    batch = itr
                    input_ids = batch["labels"].to(device)
                    prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                    loss = arm_loss(model, input_ids, eos_id=eos_id, prompt_mask=prompt_mask)
                else:
                    raise ValueError(f"Invalid training strategy: {strategy}")
            
            optimizer.zero_grad()

            loss.backward()
            if train_cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            if train_cfg.ema is not None:
                ema.update(ema_params)
            scheduler.step()
            global_step += 1
            
            # update a new seq
            if strategy in ["progressive", "progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v5", "edit_v6", "edit_v7"]:
                pool.update_with_logits(log_probs)

            if is_main:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

                if global_step % train_cfg.logging_steps == 0:
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss {loss.item()}")
                    if cfg.wandb.wandb:
                        wandb.log({"train/loss": loss.item()}, step=global_step)

                        gn = grad_norm(model.parameters())
                        wandb.log({"train/grad_norm": gn}, step=global_step)

                        if strategy in ["progressive", "progressive_edit", "edit_v2", "edit_v3", "edit_v3_2", "edit_v3_3", "edit_v4", "edit_v5", "edit_v6", "edit_v7"]:
                            wandb.log({"train/current_k": current_k}, step=global_step)

                        wandb.log({"train/lr": optimizer.param_groups[0]["lr"]}, step=global_step)

            if global_step % train_cfg.eval_steps == 0:
                model.eval()

                # validaton on the downstream task; disabled when we use EMA
                if train_cfg.ema is None:
                    val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size, step=global_step, logdir=track_dir)
                else:
                    val_acc_dict = None

                # validation loss (mdm loss on the validation dataset)
                val_loss = val_loss_ddp(model, val_loader, mask_id, device, rank, world_size, strategy, eos_id, arm_init=model_config.predict_next_token)

                # EMA evaluation
                if train_cfg.ema is not None:
                    torch.cuda.empty_cache()
                    model_to_ema = model.module if isinstance(model, DDP) else model
                    ema.store(model_to_ema.parameters())
                    ema.copy_to(model_to_ema.parameters())
                   
                    with torch.inference_mode():
                        # validaton on the downstream task
                        val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size, step=global_step, logdir=track_dir)
                    ema.restore(model_to_ema.parameters())
                
                if is_main:
                    # eval acc logging
                    for key, value in val_acc_dict.items():
                        print(f"Epoch {epoch+1}, Step {global_step}, Validation Accuracy {key}: {value}")
                        if cfg.wandb.wandb:
                            if train_cfg.ema is not None:
                                wandb.log({"ema_test_acc/" + key: value}, step=global_step)
                            else:
                                wandb.log({"test_acc/" + key: value}, step=global_step)
                    
                    # validation loss logging
                    print(f"Epoch {epoch+1}, Step {global_step}, Validation Loss: {val_loss}")
                    if cfg.wandb.wandb:
                        wandb.log({"val/val_loss": val_loss}, step=global_step)

                    if is_main and global_step % train_cfg.save_steps == 0 and train_cfg.ema is not None:
                        saved_path = save_ema_snapshot(
                            ckpt_dir, model, ema, cfg, epoch, global_step, val_loss, val_acc_dict,
                            optimizer=optimizer, scheduler=scheduler,
                        )
                        if saved_path is not None:
                            print(f"EMA Model saved to: {saved_path}")

                    if is_main and global_step % train_cfg.save_steps == 0:
                        # save non-EMA snapshot with full training state for resume
                        saved_path = save_model_snapshot(
                            ckpt_dir, model, cfg, epoch, global_step,
                            val_loss=val_loss,
                            extra=val_acc_dict,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            ema=ema if train_cfg.ema is not None else None,
                        )
                        if saved_path is not None:
                            print(f"Model saved to: {saved_path}")
                
                model.train()

            # Check if we've reached max_steps after eval
            if global_step >= num_training_steps:
                break

        # Check if we've reached max_steps after completing an epoch
        if global_step >= num_training_steps:
            break

    if cfg.wandb.wandb and is_main:
        wandb.finish()
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # args = parse_args()
    # cfg_path = args.cfg
    # cfg = OmegaConf.load(cfg_path)
    # main(cfg)
    main()
