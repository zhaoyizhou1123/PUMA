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
from progressive import PhasedMaskingEdit, mdm_edit_loss_fn, mdm_loss_fn
from eval.sudoku_eval import evaluate_ddp_sudoku
from eval.gsm8k_eval import evaluate_ddp_gsm8k

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
                sampling_no_edit.confidence = confidence
                sampling_no_edit.unmasking_num = unmasking_num
                sampling_no_edit.edit_freq = -1
                if hasattr(base_sampling, "edit_step") and len(list(base_sampling.edit_step)) > 0:
                    sampling_no_edit.edit_step = list(base_sampling.edit_step)[0]

                metric_name_no_edit = f"{confidence}_unmasking_{unmasking_num}_editfreq_-1"
                if hasattr(sampling_no_edit, 'edit_step'):
                    metric_name_no_edit += f"_editstep_{sampling_no_edit.edit_step}"
                result_no_edit = evaluate_ddp(model, cfg, device, rank, world_size, sampling_no_edit, step=step, logdir=logdir, metric_name=metric_name_no_edit)

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
                        out[metric_name] = evaluate_ddp(model, cfg, device, rank, world_size, sampling, step=step, logdir=logdir, metric_name=metric_name)
    return out

def evaluate_ddp(model, cfg, device, rank: int, world_size: int, sampling, step=0, logdir=None, metric_name=""):
    if cfg.data.dataset == "sudoku":
        return evaluate_ddp_sudoku(model, cfg, device, rank, world_size, sampling, step=step, logdir=logdir)
    elif cfg.data.dataset == "tinygsm":
        return evaluate_ddp_gsm8k(model, cfg, device, rank, world_size, sampling)
    elif cfg.data.dataset == "maze":
        from eval.maze_eval import evaluate_ddp_maze
        return evaluate_ddp_maze(model, cfg, device, rank, world_size, sampling, step=step, logdir=logdir, metric_name=metric_name)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")

@hydra.main(version_base=None, config_path="../yaml_files/maze", config_name="maze")
def main(cfg: DictConfig):
    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start training!")
        print(f"Training with {world_size} GPUs")

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

    data_cfg = cfg.data
    train_cfg = cfg.training

    # Create logdir
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join("track", cfg.wandb.project, cfg.wandb.name, datetime_str)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize the model
    model_cfg_dict = cfg.model
    model_config = MDMConfig(**model_cfg_dict)
    model = MDMTransformer(model_config).to(device)

    assert 'ckpt_path' in cfg.validation, "ckpt_path must be specified in the config for evaluation"
    print(f"Loading checkpoint from {cfg.validation.ckpt_path} for evaluation")
    ckpt = torch.load(cfg.validation.ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)

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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"Model wrapping is done!")

    model.eval()

    val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size, logdir=logdir)

    if is_main:
        # eval acc logging
        for key, value in val_acc_dict.items():
            print(f"Validation Accuracy {key}: {value}")
                
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # args = parse_args()
    # cfg_path = args.cfg
    # cfg = OmegaConf.load(cfg_path)
    # main(cfg)
    main()
