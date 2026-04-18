"""
BackPlay fine-tuning training script
=====================================
Usage (single GPU):
    torchrun --nproc_per_node=1 -m backplay.train \\
        --config-path ../yaml_files/sudoku_hard \\
        --config-name backplay_finetune \\
        backplay.pretrained_ckpt=/path/to/step10000.pt

The backbone is loaded from backplay.pretrained_ckpt and frozen by default.
Only the correction head is trained.

Config sections required (beyond the base model/data/validation/wandb fields):
    backplay:
      pretrained_ckpt: "path/to/backbone.pt"
      delta_t: 0.1                 # minimum look-back distance for LBC
      reg_lambda: 0.0
      tune_backbone: false
      adapter_layers: 2
      tau: 0.5                     # correction threshold (inference)
      remask_budget: 2             # K in Algorithm 1
      stride: 1                    # correct every d steps (inference)
      block_size: 10               # Hblock size (inference)
"""

from __future__ import annotations
import datetime
import math
import os
import random
import sys
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data import setup_data_bundle
from data.sudoku_utils import resolve_sudoku_grid_size
from eval.sudoku_eval import evaluate_ddp_sudoku
from model.ema import save_model_snapshot
from model.transformer import MDMConfig
from backplay.algorithm import backplay_training_step
from backplay.model import BackPlayMDMTransformer
from backplay.sampling import backplay_evaluate_ddp_sudoku, backplay_sampling
from sampling import mdm_sampling


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total ** 0.5


def compute_scaled_lr(base_lr: float, batch_size: int, mode: str, base_bs: int = 128) -> float:
    if mode == "constant":
        return base_lr
    elif mode == "sqrt":
        return base_lr * math.sqrt(batch_size / base_bs)
    elif mode == "linear":
        return base_lr * (batch_size / base_bs)
    raise ValueError(f"Unknown lr_scaling_mode: {mode}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def val_loss_ddp(model, val_loader, mask_id: int, device, rank: int, world_size: int):
    """Standard MDM CE loss on validation set — monitors backbone health."""
    model.eval()
    local_sum, local_count = 0.0, 0

    if world_size > 1 and dist.is_initialized():
        sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
            val_loader.dataset, batch_size=val_loader.batch_size or 16,
            sampler=sampler, num_workers=0, pin_memory=False, drop_last=False,
        )

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val loss", disable=(rank != 0)):
            x0 = batch["labels"].to(device)
            pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
            B, L = x0.shape
            if pm is None:
                pm = torch.zeros_like(x0, dtype=torch.bool)
            mask_prob = torch.rand(B, 1, device=device)
            rand_mask = torch.rand(B, L, device=device)
            mask_idx = (rand_mask < mask_prob) & (~pm)
            x_t = torch.where(mask_idx, mask_id, x0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(x_t)
            loss = F.cross_entropy(logits[mask_idx], x0[mask_idx])
            local_sum   += loss.item() * B
            local_count += B

    tensor = torch.tensor([local_sum, float(local_count)], device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor[0] / tensor[1].clamp(min=1)).item()


def evaluate(model, cfg, device, rank, world_size, sampling_cfg, step, logdir, use_backplay=False):
    """Run sudoku solve-accuracy eval using either standard or BackPlay sampling."""
    if use_backplay:
        return backplay_evaluate_ddp_sudoku(model, cfg, device, rank, world_size, sampling_cfg, step=step, logdir=logdir)
    else:
        return evaluate_ddp_sudoku(model, cfg, device, rank, world_size, sampling_cfg, step=step, logdir=logdir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../yaml_files/sudoku_hard", config_name="backplay_finetune")
def main(cfg: DictConfig):
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    if is_main:
        print("BackPlay fine-tuning — starting")
        print(f"World size: {world_size}")

    # Resolve grid_size → vocab_size / max_position / mask_id for sudoku
    cfg = resolve_sudoku_grid_size(cfg)

    # Per-GPU batch size
    bs = cfg.training.batch_size
    assert bs % world_size == 0, f"batch_size {bs} must be divisible by world_size {world_size}"
    with open_dict(cfg):
        cfg.data.training.per_gpu_batch_size = bs // world_size

    # Reproducibility
    seed = cfg.data.seed + rank
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    ts       = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    ckpt_root = cfg.training.get("ckpt_root", "ckpts")
    ckpt_dir  = f"{ckpt_root}/{cfg.wandb.project}/{cfg.wandb.name}_date{ts}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints: {ckpt_dir}")

    track_dir = None
    if cfg.validation.get("track", False):
        track_dir = f"track/backplay/date={ts}"
        os.makedirs(track_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model: build BackPlayMDMTransformer, load backbone from checkpoint
    # ------------------------------------------------------------------
    model_cfg     = cfg.model
    backplay_cfg  = cfg.backplay
    model_config  = MDMConfig(**{k: v for k, v in model_cfg.items()})

    adapter_layers = int(backplay_cfg.get("adapter_layers", 2))
    model = BackPlayMDMTransformer(model_config, adapter_layers=adapter_layers).to(device)

    # Load pretrained backbone weights
    pretrained_ckpt = backplay_cfg.pretrained_ckpt
    if pretrained_ckpt and pretrained_ckpt != "none":
        if is_main:
            print(f"Loading backbone from: {pretrained_ckpt}")
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        sd   = ckpt.get("model_state_dict", ckpt)
        # Drop adapter keys if present in checkpoint (fresh adapter every run)
        sd   = {k: v for k, v in sd.items() if not k.startswith("adapter.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"  missing keys : {missing}")
            print(f"  unexpected   : {unexpected}")
    else:
        if is_main:
            print("WARNING: no pretrained_ckpt provided — backbone is randomly initialised!")

    # Freeze backbone, only train adapter
    tune_backbone = bool(backplay_cfg.get("tune_backbone", False))
    for name, param in model.named_parameters():
        if name.startswith("adapter."):
            param.requires_grad = True
        else:
            param.requires_grad = tune_backbone

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        total  = sum(p.numel() for p in model.parameters())
        adapt  = sum(p.numel() for p in trainable_params)
        print(f"Total params: {total/1e6:.2f}M  |  Trainable (adapter): {adapt/1e6:.3f}M")

    # ------------------------------------------------------------------
    # DDP wrap
    # ------------------------------------------------------------------
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_bundle  = setup_data_bundle(cfg.data)
    train_loader = data_bundle.train_loader
    val_loader   = data_bundle.val_loader
    mask_id      = cfg.data.mask_id

    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader  = DataLoader(
            train_loader.dataset,
            batch_size=cfg.data.training.per_gpu_batch_size,
            sampler=train_sampler,
            num_workers=0, pin_memory=False, drop_last=False,
        )
    else:
        train_sampler = None

    # ------------------------------------------------------------------
    # Optimiser + scheduler  (adapter params only, or all if tune_backbone)
    # ------------------------------------------------------------------
    train_cfg   = cfg.training
    lr_mode     = getattr(train_cfg, "lr_scaling_mode", "constant")
    scaled_lr   = compute_scaled_lr(train_cfg.learning_rate, train_cfg.batch_size, lr_mode)
    if is_main:
        print(f"LR scaling: {lr_mode}  base={train_cfg.learning_rate}  scaled={scaled_lr:.2e}")

    optimizer = optim.AdamW(trainable_params, lr=scaled_lr, weight_decay=train_cfg.weight_decay)

    num_training_steps = getattr(train_cfg, "max_steps", None)
    if num_training_steps is None:
        num_training_steps = train_cfg.num_epochs * len(train_loader)
    max_epochs = max(1, math.ceil(num_training_steps / len(train_loader)))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps
    )

    if is_main:
        print(f"Training for {num_training_steps} steps (~{max_epochs} epochs)")

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    if cfg.wandb.wandb and is_main:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    bp_delta_t    = float(backplay_cfg.delta_t)
    bp_reg_lambda = float(backplay_cfg.reg_lambda)
    bp_error_weight = float(backplay_cfg.get("error_weight", 1.0))

    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(not is_main))

        for batch in pbar:
            if global_step >= num_training_steps:
                break

            x0 = batch["labels"].to(device)
            pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else \
                 torch.zeros_like(x0, dtype=torch.bool)

            # Unwrap model for direct call inside backplay_training_step
            raw_model = model.module if isinstance(model, DDP) else model

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                result = backplay_training_step(
                    raw_model, x0, pm, mask_id,
                    delta_t=bp_delta_t,
                    reg_lambda=bp_reg_lambda,
                    tune_backbone=tune_backbone,
                    error_weight=bp_error_weight,
                )

            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            if train_cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable_params, train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                if global_step % train_cfg.logging_steps == 0:
                    log = {
                        "train/loss":    loss.item(),
                        "train/loss_sc": result["loss_sc"].item(),
                        "train/lr":      optimizer.param_groups[0]["lr"],
                    }
                    if "loss_reg" in result:
                        log["train/loss_reg"] = result["loss_reg"].item()
                    if cfg.wandb.wandb:
                        wandb.log(log, step=global_step)
                    else:
                        print(f"Step {global_step}  " + "  ".join(f"{k}={v:.4f}" for k, v in log.items()))

            # ---- Evaluation ----
            if global_step % train_cfg.eval_steps == 0:
                model.eval()

                # 1. Backbone val loss
                v_loss = val_loss_ddp(model, val_loader, mask_id, device, rank, world_size)

                # 2. Standard sampling accuracy (baseline — no adapter)
                sampling_cfg = deepcopy(cfg.validation.sampling)
                std_acc = evaluate(model, cfg, device, rank, world_size, sampling_cfg, global_step, track_dir, use_backplay=False)

                # 3. BackPlay sampling accuracy (adapter-guided remasking)
                bp_sampling_cfg = deepcopy(cfg.validation.sampling)
                with open_dict(bp_sampling_cfg):
                    bp_sampling_cfg.tau        = float(backplay_cfg.get("tau",        0.5))
                    bp_sampling_cfg.remask_budget = int(backplay_cfg.get("remask_budget", 2))
                    bp_sampling_cfg.stride     = int(backplay_cfg.get("stride",     1))
                    bp_sampling_cfg.block_size = int(backplay_cfg.get("block_size",  10))
                bp_eval = evaluate(model, cfg, device, rank, world_size, bp_sampling_cfg, global_step, track_dir, use_backplay=True)
                bp_acc = bp_eval["acc"]

                if is_main:
                    print(
                        f"Step {global_step}  val_loss={v_loss:.4f}  std_acc={std_acc:.4f}  "
                        f"backplay_acc={bp_acc:.4f}  remask_seq_frac={bp_eval['remask_seq_frac']:.4f}  "
                        f"avg_remasked={bp_eval['avg_remasked_tokens_per_seq']:.4f}  "
                        f"avg_err={bp_eval['avg_error_prob']:.4f}  "
                        f"avg_max_err={bp_eval['avg_max_error_prob_per_correction']:.4f}"
                    )
                    if cfg.wandb.wandb:
                        wandb.log({
                            "val/val_loss":     v_loss,
                            "val/std_acc":      std_acc,
                            "val/backplay_acc": bp_acc,
                            "val/remask_seq_frac": bp_eval["remask_seq_frac"],
                            "val/avg_remasked_tokens_per_seq": bp_eval["avg_remasked_tokens_per_seq"],
                            "val/avg_error_prob": bp_eval["avg_error_prob"],
                            "val/avg_max_error_prob_per_correction": bp_eval["avg_max_error_prob_per_correction"],
                            "val/avg_backplay_steps_per_batch": bp_eval["avg_steps_per_batch"],
                            "val/avg_backplay_correction_steps_per_batch": bp_eval["avg_correction_steps_per_batch"],
                        }, step=global_step)

                # ---- Checkpoint ----
                if is_main and global_step % train_cfg.save_steps == 0:
                    saved = save_model_snapshot(
                        ckpt_dir, model, cfg, epoch, global_step,
                        val_loss=v_loss,
                        extra={"std_acc": std_acc, "backplay_acc": bp_acc},
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    if saved:
                        print(f"Checkpoint saved: {saved}")

                model.train()

        if global_step >= num_training_steps:
            break

    if cfg.wandb.wandb and is_main:
        wandb.finish()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
