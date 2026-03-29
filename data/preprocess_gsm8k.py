import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
INPUT_FILE = "/home/zhaoyiz/projects/SMDM/data/gsm8k/train.txt"
OUT_DIR = os.path.join(os.path.dirname(__file__), "gsm8k")

Q_MAX_LEN = 128   # left-padded with bos to this length
A_MAX_LEN = 64    # right-padded with eos to this length
SEQ_LEN = Q_MAX_LEN + A_MAX_LEN


def preprocess_gsm8k(
    input_file: str = INPUT_FILE,
    out_dir: str = OUT_DIR,
    q_max_len: int = Q_MAX_LEN,
    a_max_len: int = A_MAX_LEN,
    tokenizer_name: str = TOKENIZER_NAME,
):
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has no bos_token_id")
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    seq_len = q_max_len + a_max_len
    all_tokens = []
    all_masks = []
    discarded = 0

    for line in tqdm(lines, desc="Tokenizing GSM8K"):
        if "||" not in line:
            discarded += 1
            continue

        question, answer = line.split("||", 1)
        question = question.strip()
        answer = answer.strip()

        q_ids = tok(question, add_special_tokens=False).input_ids
        a_ids = tok(answer, add_special_tokens=False).input_ids

        # Discard if either part exceeds max length
        if len(q_ids) > q_max_len or len(a_ids) > a_max_len:
            discarded += 1
            continue

        # Left-pad question with bos to q_max_len
        q_pad_len = q_max_len - len(q_ids)
        q_padded = [bos_id] * q_pad_len + q_ids

        # Right-pad answer with eos to a_max_len
        a_pad_len = a_max_len - len(a_ids)
        a_padded = a_ids + [eos_id] * a_pad_len

        # Concatenate
        seq = q_padded + a_padded  # length = seq_len

        # Prompt mask: True for question positions, False for answer positions
        prompt_mask = [True] * q_max_len + [False] * a_max_len

        all_tokens.append(seq)
        all_masks.append(prompt_mask)

    tokens_arr = np.array(all_tokens, dtype=np.int32)   # [N, seq_len]
    masks_arr = np.array(all_masks, dtype=bool)          # [N, seq_len]

    tokens_path = os.path.join(out_dir, "train_label.npy")
    mask_path = os.path.join(out_dir, "train_prompt_mask.npy")

    np.save(tokens_path, tokens_arr)
    np.save(mask_path, masks_arr)

    print(f"Done. Kept {len(all_tokens):,} examples, discarded {discarded:,}.")
    print(f"Sequence length: {seq_len} ({q_max_len} question + {a_max_len} answer)")
    print(f"- {tokens_path}  shape={tokens_arr.shape}  dtype={tokens_arr.dtype}")
    print(f"- {mask_path}  shape={masks_arr.shape}  dtype={masks_arr.dtype}")


if __name__ == "__main__":
    preprocess_gsm8k()
