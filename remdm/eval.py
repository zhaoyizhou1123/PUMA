"""
ReMDM evaluation script — no training required.

Loads a pretrained MDM checkpoint and evaluates solve accuracy using the
ReMDM remasking sampler across the configured sweep. `conf` is evaluated once,
while `cap` and `rescale` are swept over the configured `eta` values.

Usage:
    torchrun --nproc_per_node=1 -m remdm.eval \\
        --cfg yaml_files/sudoku_hard/remdm_eval.yaml \\
        --ckpt /path/to/step_N.pt
"""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from model.transformer import MDMConfig, MDMTransformer
from remdm.sampling import remdm_evaluate_ddp_sudoku


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",  type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to pretrained MDM checkpoint (.pt)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.cfg)

    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    seed = cfg.data.seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ---- load pretrained MDM ----
    model_config = MDMConfig(**cfg.model)
    model        = MDMTransformer(model_config).to(device)
    ckpt         = torch.load(args.ckpt, map_location="cpu")
    sd           = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded: {args.ckpt}")
        print(f"Params: {n_params / 1e6:.2f}M")

    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ---- sweep over variants x etas ----
    remdm_cfg     = cfg.remdm
    variants      = list(remdm_cfg.variants)
    etas          = list(remdm_cfg.etas)
    base_sampling = cfg.validation.sampling

    results = {}
    for variant in variants:
        variant_etas = etas if variant != "conf" else [None]
        for eta in variant_etas:
            sampling = deepcopy(base_sampling)
            with open_dict(sampling):
                sampling.variant = variant
                if eta is not None:
                    sampling.eta = float(eta)

            acc = remdm_evaluate_ddp_sudoku(
                model, cfg, device, rank, world_size, sampling
            )
            key = (
                f"variant={variant}"
                if eta is None else
                f"variant={variant}  eta={eta:.3f}"
            )
            results[key] = acc
            if is_main:
                print(f"{key}  ->  acc={acc:.4f}")

    if is_main:
        print("\n--- Summary ---")
        for key, acc in results.items():
            print(f"  {key}  acc={acc:.4f}")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
