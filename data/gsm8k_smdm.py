import numpy as np
import torch
from torch.utils.data import Dataset


class GSM8KSMDMDataset(Dataset):
    """
    Wraps pre-tokenized GSM8K numpy arrays.
    Returns {"labels": LongTensor[seq_len], "prompt_mask": BoolTensor[seq_len]}
    """
    def __init__(self, data, prompt_mask):
        self.data = data
        self.prompt_mask = prompt_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"labels": self.data[idx], "prompt_mask": self.prompt_mask[idx]}


def split_gsm8k_smdm(data_dir: str, val_ratio: float = 0.02, seed: int = 123):
    import os
    data = torch.from_numpy(np.load(os.path.join(data_dir, "train_label.npy"))).long()
    prompt_mask = torch.from_numpy(np.load(os.path.join(data_dir, "train_prompt_mask.npy"))).bool()

    N = len(data)
    n_val = max(1, int(N * val_ratio))
    n_train = N - n_val
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=gen)
    train_ds = GSM8KSMDMDataset(data[idx[:n_train]], prompt_mask[idx[:n_train]])
    val_ds = GSM8KSMDMDataset(data[idx[n_train:]], prompt_mask[idx[n_train:]])
    return train_ds, val_ds
