"""
RemeDi SFT fine-tuning training script
========================================
Usage (single GPU):
    torchrun --nproc_per_node=1 -m remedi.train \\
        --config-path ../yaml_files/sudoku_hard \\
        --config-name remedi_finetune \\
        remedi.pretrained_ckpt=/path/to/step_N.pt

Config sections required (beyond the base model/data/validation/wandb fields):
    remedi:
      pretrained_ckpt: "path/to/backbone.pt"
      ups_positions: [1, 3, 5, 7]   # UPS tap layers (0-indexed)
      incorrect_ratio: 0.1
      lambda_ups: 0.3
      tune_backbone: true
      backprop_warmup_steps: 0      # (non-paper) freeze backprop_linears for first N steps; 0 = disabled
      # inference settings
      use_ups: false                 # false = GitHub TPS gather; true = UPS ranking
"""

from __future__ import annotations

import datetime
import math
import os
import random
import sys

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
from remedi.algorithm import remedi_training_step
from remedi.model import RemeDiMDMTransformer
from remedi.sampling import remedi_evaluate_ddp_sudoku


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
        sampler    = DistributedSampler(
            val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        val_loader = DataLoader(
            val_loader.dataset, batch_size=val_loader.batch_size or 16,
            sampler=sampler, num_workers=0, pin_memory=False, drop_last=False,
        )

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val loss", disable=(rank != 0)):
            x0 = batch["labels"].to(device)
            pm = (batch["prompt_mask"].to(device) if "prompt_mask" in batch
                  else torch.zeros_like(x0, dtype=torch.bool))
            B, L    = x0.shape
            L_eff   = (~pm).sum(dim=1, keepdim=True).clamp(min=1).float()
            num_mask = (torch.rand(B, 1, device=device) * L_eff).long().clamp(min=1)
            noise   = torch.rand(B, L, device=device).masked_fill(pm, float('inf'))
            order   = noise.argsort(dim=1)
            mask_idx = order < num_mask
            x_t     = torch.where(mask_idx, mask_id, x0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                logits = model(x_t)
            loss = F.cross_entropy(logits[mask_idx], x0[mask_idx])
            local_sum   += loss.item() * B
            local_count += B

    tensor = torch.tensor([local_sum, float(local_count)], device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor[0] / tensor[1].clamp(min=1)).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../yaml_files/sudoku_hard", config_name="remedi_finetune")
def main(cfg: DictConfig):
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    if is_main:
        print("RemeDi SFT fine-tuning — starting")
        print(f"World size: {world_size}")

    cfg = resolve_sudoku_grid_size(cfg)

    bs = cfg.training.batch_size
    assert bs % world_size == 0, f"batch_size {bs} must be divisible by world_size {world_size}"
    with open_dict(cfg):
        cfg.data.training.per_gpu_batch_size = bs // world_size

    seed = cfg.data.seed + rank
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = (torch.device(f"cuda:{local_rank}") if torch.cuda.is_available()
              else torch.device("cpu"))

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    ts        = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    ckpt_root = cfg.training.get("ckpt_root", "ckpts")
    ckpt_dir  = f"{ckpt_root}/{cfg.wandb.project}/{cfg.wandb.name}_date{ts}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Model: RemeDiMDMTransformer
    # ------------------------------------------------------------------
    model_cfg    = cfg.model
    remedi_cfg   = cfg.remedi
    model_config = MDMConfig(**{k: v for k, v in model_cfg.items()})

    ups_positions = tuple(remedi_cfg.get("ups_positions", [1, 3, 5, 7]))
    model = RemeDiMDMTransformer(model_config, ups_positions=ups_positions).to(device)

    # Load pretrained backbone weights (strict=False: UPS keys are new)
    pretrained_ckpt = remedi_cfg.pretrained_ckpt
    if pretrained_ckpt and pretrained_ckpt != "none":
        if is_main:
            print(f"Loading backbone from: {pretrained_ckpt}")
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        sd   = ckpt.get("model_state_dict", ckpt)
        # Drop any UPS keys that might be present in the checkpoint
        ups_prefixes = ("ups_blocks.", "backprop_linears.", "backprop_norms.",
                        "ups_final_norm.", "confidence_head.")
        sd = {k: v for k, v in sd.items()
              if not any(k.startswith(p) for p in ups_prefixes)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"  missing keys  : {missing}")
            print(f"  unexpected    : {unexpected}")
    else:
        if is_main:
            print("WARNING: no pretrained_ckpt — backbone is randomly initialised!")

    # Freeze/unfreeze backbone
    tune_backbone = bool(remedi_cfg.get("tune_backbone", True))
    ups_prefixes  = ("ups_blocks.", "backprop_linears.", "backprop_norms.",
                     "ups_final_norm.", "confidence_head.")
    # (non-paper) optionally freeze backprop_linears for the first N steps so that
    # random UPS write-back does not corrupt the pretrained TPS before UPS has learned
    # anything useful. Set backprop_warmup_steps=0 (default) to disable entirely.
    backprop_warmup_steps = int(remedi_cfg.get("backprop_warmup_steps", 0))
    for name, param in model.named_parameters():
        is_ups = any(name.startswith(p) for p in ups_prefixes)
        if backprop_warmup_steps > 0 and name.startswith("backprop_linears."):
            param.requires_grad = False   # temporarily frozen; unfrozen at step N
        else:
            param.requires_grad = True if is_ups else tune_backbone

    if is_main:
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total/1e6:.2f}M  |  Trainable: {train/1e6:.3f}M")
        if backprop_warmup_steps > 0:
            print(f"backprop_linears frozen for first {backprop_warmup_steps} steps")

    # ------------------------------------------------------------------
    # DDP
    # ------------------------------------------------------------------
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_bundle  = setup_data_bundle(cfg.data)
    train_loader = data_bundle.train_loader
    val_loader   = data_bundle.val_loader
    mask_id      = cfg.data.mask_id

    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=cfg.data.training.per_gpu_batch_size,
            sampler=train_sampler,
            num_workers=0, pin_memory=False, drop_last=False,
        )
    else:
        train_sampler = None

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # Two separate learning rates per the paper:
    #   UPS params: remedi.ups_lr   (default 2e-5, or tps_lr×10 if unset)
    #   TPS params: remedi.tps_lr   (default 2e-6)
    # ------------------------------------------------------------------
    train_cfg = cfg.training
    tps_lr    = float(remedi_cfg.get("tps_lr", train_cfg.learning_rate))
    ups_lr    = float(remedi_cfg.get("ups_lr", tps_lr * 10.0))
    if is_main:
        print(f"LR — TPS: {tps_lr:.2e}  UPS: {ups_lr:.2e}")

    ups_pfx_set = set(ups_prefixes)
    ups_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and any(n.startswith(px) for px in ups_prefixes)]
    tps_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not any(n.startswith(px) for px in ups_prefixes)]

    optimizer = optim.AdamW(
        [
            {"params": ups_params, "lr": ups_lr},
            {"params": tps_params, "lr": tps_lr},
        ],
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    num_training_steps = getattr(train_cfg, "max_steps", None)
    if num_training_steps is None:
        num_training_steps = train_cfg.num_epochs * len(train_loader)
    max_epochs = max(1, math.ceil(num_training_steps / len(train_loader)))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )

    if is_main:
        print(f"Training for {num_training_steps} steps (~{max_epochs} epochs)")

    # ------------------------------------------------------------------
    # WandB (SLURM/NFS: disable background service — avoids ServicePollForTokenError)
    # ------------------------------------------------------------------
    if os.environ.get("SLURM_JOB_ID") and os.environ.get(
        "WANDB_DISABLE_SERVICE", ""
    ).lower() not in ("0", "false", "no"):
        os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

    use_wandb = bool(cfg.wandb.wandb)
    if use_wandb and is_main:
        try:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            print(f"Warning: wandb.init failed ({e!r}); continuing without W&B.", flush=True)
            use_wandb = False

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    remedi_incorrect_ratio = float(remedi_cfg.get("incorrect_ratio", 0.1))
    remedi_lambda_ups = float(remedi_cfg.get("lambda_ups", 0.3))

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
            pm = (batch["prompt_mask"].to(device) if "prompt_mask" in batch
                  else torch.zeros_like(x0, dtype=torch.bool))

            raw_model = model.module if isinstance(model, DDP) else model

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                result = remedi_training_step(
                    raw_model, x0, pm, mask_id,
                    vocab_size=model_config.vocab_size,
                    incorrect_ratio=remedi_incorrect_ratio,
                    lambda_ups=remedi_lambda_ups,
                    detach_tps=(backprop_warmup_steps > 0 and global_step < backprop_warmup_steps),
                    skip_diffusion=(backprop_warmup_steps > 0 and global_step < backprop_warmup_steps),
                )

            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            if train_cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.max_grad_norm,
                )
            optimizer.step()
            scheduler.step()
            global_step += 1

            # (non-paper) unfreeze backprop_linears once UPS has warmed up
            if backprop_warmup_steps > 0 and global_step == backprop_warmup_steps:
                new_params = [
                    p for n, p in raw_model.named_parameters()
                    if n.startswith("backprop_linears.")
                ]
                for p in new_params:
                    p.requires_grad = True
                optimizer.add_param_group({"params": new_params, "lr": ups_lr})
                if is_main:
                    print(f"Step {global_step}: unfreezing backprop_linears")

            if is_main:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

                if global_step % train_cfg.logging_steps == 0:
                    log = {
                        "train/loss":           loss.item(),
                        "train/loss_diffusion": result["loss_diffusion"].item(),
                        "train/loss_ups":       result["loss_ups"].item(),
                        "train/lr":             optimizer.param_groups[0]["lr"],
                    }
                    if use_wandb:
                        wandb.log(log, step=global_step)
                    else:
                        print(f"Step {global_step}  "
                              + "  ".join(f"{k}={v:.4f}" for k, v in log.items()))

            # ---- Evaluation ----
            if global_step % train_cfg.eval_steps == 0:
                model.eval()

                # 1. Backbone val loss
                v_loss = val_loss_ddp(model, val_loader, mask_id, device, rank, world_size)

                # 2. Standard sampling accuracy (no UPS guidance)
                std_acc = evaluate_ddp_sudoku(
                    model, cfg, device, rank, world_size,
                    deepcopy(cfg.validation.sampling),
                    step=global_step, logdir=None,
                )

                # 3. RemeDi global top-K sampling accuracy
                remedi_sampling_cfg = deepcopy(cfg.validation.sampling)
                with open_dict(remedi_sampling_cfg):
                    remedi_sampling_cfg.use_ups = bool(remedi_cfg.get("use_ups", False))
                remedi_acc = remedi_evaluate_ddp_sudoku(
                    raw_model, cfg, device, rank, world_size,
                    remedi_sampling_cfg, step=global_step, logdir=None,
                )

                if is_main:
                    print(f"Step {global_step}  val_loss={v_loss:.4f}"
                          f"  std_acc={std_acc:.4f}  remedi_acc={remedi_acc:.4f}")
                    if use_wandb:
                        wandb.log({
                            "val/val_loss":   v_loss,
                            "val/std_acc":    std_acc,
                            "val/remedi_acc": remedi_acc,
                        }, step=global_step)

                # ---- Checkpoint ----
                if is_main and global_step % train_cfg.save_steps == 0:
                    saved = save_model_snapshot(
                        ckpt_dir, model, cfg, epoch, global_step,
                        val_loss=v_loss,
                        extra={"std_acc": std_acc, "remedi_acc": remedi_acc},
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    if saved:
                        print(f"Checkpoint saved: {saved}")

                model.train()

        if global_step >= num_training_steps:
            break

    if use_wandb and is_main:
        wandb.finish()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
