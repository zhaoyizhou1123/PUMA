# Maze representation from MaxRL paper.

import json, os
import numpy as np
from tqdm import tqdm

def tokenize_maze(data_item: dict, vocab_map: dict, max_path_length: int):
    seq = data_item["sequence"].split()
    seq = seq[1:-1] # remove <bos> and <eos>
    opt_len = data_item['optimal_path_length']
    if opt_len > max_path_length:
        return None, None
    else:
        pad_len = max_path_length - opt_len
        pad_list = ["DONE"] * pad_len
        seq = seq + pad_list
    tokenized_seq = [vocab_map[token] for token in seq]

    # get prompt length
    prompt_length = seq.index("PATH_START") + 1
    prompt_mask = np.zeros(len(tokenized_seq), dtype=bool)
    prompt_mask[:prompt_length] = 1

    return tokenized_seq, prompt_mask


def process_file(input_file: str, save_dir: str, split: str, vocab_map: dict, max_path_length: int):
    seq_list = []
    mask_list = []
    with open(input_file, "r") as fr:
        for line in tqdm(fr):
            data_item = json.loads(line)
            tokenized_seq, prompt_mask = tokenize_maze(data_item, vocab_map, max_path_length)
            if tokenized_seq is None:
                continue
            seq_list.append(tokenized_seq)
            mask_list.append(prompt_mask)
    seq_array = np.array(seq_list)
    mask_arr = np.stack(mask_list, axis=0)
    print(seq_array.shape, mask_arr.shape)
    print(mask_arr[0].sum())
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{split}_labels.npy"), seq_array)
    np.save(os.path.join(save_dir, f"{split}_prompt_mask.npy"), mask_arr)

VOCAB_MAP = {
      "GRID_START": 1,
      "GRID_END": 2,
      "PATH_START": 3,
      "DONE": 4,
      "PATH": 5,
      "WALL": 6,
      "GOAL": 7,
      "START": 8,
      "NEWLINE": 9,
      "UP": 10,
      "DOWN": 11,
      "LEFT": 12,
      "RIGHT": 13,
}

if __name__ == "__main__":
    source_dir = "data/maze17x17_dfs_v2"
    train_file = os.path.join(source_dir, "train.jsonl")
    test_file = os.path.join(source_dir, "test.jsonl")
    target_dir = "data/maze17x17_dfs_v2"
    process_file(test_file, target_dir, "test", VOCAB_MAP, max_path_length=100)
    process_file(train_file, target_dir, "train", VOCAB_MAP, max_path_length=100)
    