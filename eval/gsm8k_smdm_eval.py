import torch
import os
import io
import contextlib
import signal
import numpy as np
import math
import torch.distributed as dist
import re
import json
import warnings
from transformers import AutoTokenizer
from sampling import mdm_sampling, mdm_sampling_block, arm_sampling
from tqdm import tqdm
from datasets import load_dataset


# -----------------------------
# Tokenizer cache (Qwen2 default)
# -----------------------------
def get_tokenizer(tokenizer_name=None):
    name = tokenizer_name
    return AutoTokenizer.from_pretrained(name, use_fast=True)

def get_sep_ids(tokenizer_name=None):
    tok = get_tokenizer(tokenizer_name)
    return tok(SEP, add_special_tokens=False).input_ids

# TOKENIZER_NAME = "Qwen/Qwen2-0.5B"
PROMPT_MAX_LEN = 128
RESPONSE_MAX_LEN = 64
SEP = "\n"
MASK_ID = 151644

# SEP__ID: 198
# PAD__ID: 151643
# EOS__ID: 151643
# model_vocab_size: 151645

# -----------------------------
# GSM8K answer parsing
# -----------------------------

_ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
def extract_gsm8k_final_answer(ans_text: str) -> str:
    """
    GSM8K 'answer' field includes reasoning and ends with: '#### 72'
    Returns the numeric string ('72', '-3', '1,234', '10.5', etc.)
    """
    m = _ANS_RE.search(ans_text)
    if not m:
        # fallback: try last number in string
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", ans_text)
        return nums[-1].replace(",", "") if nums else ""
    return m.group(1).replace(",", "")

def test_gsm8k_tokenization(mask_id: int, tokenizer_name: str = None, data_path: str = None):
    """
    Creates/loads a tokenized GSM8K test cache.
    Cache path is keyed by tokenizer name (and data source) to avoid collisions.

    If data_path is provided, loads from a .jsonl file with fields:
        question, target  (SMDM format; prompt prefixed with "Question: ")
    Otherwise, downloads from openai/gsm8k via HuggingFace datasets.

    Returns:
        X: np.ndarray[num_test, PROMPT_MAX_LEN + RESPONSE_MAX_LEN]
        answers: list[num_test]
    """
    prompt_max = PROMPT_MAX_LEN
    resp_max = RESPONSE_MAX_LEN
    tok_tag = (tokenizer_name).replace("/", "_")
    src_tag = os.path.basename(data_path).replace(".", "_") if data_path else "hf"
    out_path = os.path.join("data", "gsm8k", f"test_mdm_{tok_tag}_{src_tag}_prompt{prompt_max}_resp{resp_max}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def load_cached():
        with open(out_path, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
        X = np.array([r["input_ids"] for r in records], dtype=np.int64)
        answers = [r["answer"] for r in records]
        return X, answers

    if os.path.exists(out_path):
        return load_cached()

    # ---- DDP-safe build: only rank0 writes ----
    ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp else 0

    if ddp and rank != 0:
        dist.barrier()
        return load_cached()

    tokenizer = get_tokenizer(tokenizer_name)
    sep_ids = get_sep_ids(tokenizer_name)
    bos_id = tokenizer.bos_token_id
    records = []

    def build_record(prompt_ids, gold):
        # Filter out examples whose prompt exceeds the max prompt length
        if len(prompt_ids) > prompt_max:
            return None
        # Left-pad prompt with BOS token to prompt_max
        pad_len = prompt_max - len(prompt_ids)
        ids = [bos_id] * pad_len + prompt_ids + [mask_id] * resp_max
        return {"input_ids": ids, "answer": gold}

    if data_path is not None:
        # SMDM-style jsonl: {"question": ..., "target": "18", ...}
        with open(data_path, "r") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        for ex in raw:
            q = (ex.get("question") or "").strip()
            gold = str(ex.get("target") or "").strip()

            q_ids = tokenizer(q, add_special_tokens=False).input_ids
            prompt_ids = q_ids

            rec = build_record(prompt_ids, gold)
            if rec is not None:
                records.append(rec)
    else:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for ex in ds:
            q = (ex.get("question") or "").strip()
            gold = extract_gsm8k_final_answer(ex.get("answer") or "")

            q_ids = tokenizer(q, add_special_tokens=False).input_ids
            prompt_ids = q_ids

            rec = build_record(prompt_ids, gold)
            if rec is not None:
                records.append(rec)

    tmp_path = out_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp_path, out_path)

    if ddp:
        dist.barrier()

    return load_cached()


def evaluate_ddp_gsm8k(model, cfg, device, rank: int, world_size: int, sampling, logdir=None, metric_name=""):
    mask_id = cfg.data.mask_id
    tokenizer_name = cfg.data.get("tokenizer_name", None)
    data_path = cfg.data.get("gsm8k_test_path", None)
    test_ratio = cfg.data.get("test_ratio", 1.0)

    # build / load the tokenized test set for this tokenizer
    X, answers = test_gsm8k_tokenization(mask_id, tokenizer_name=tokenizer_name, data_path=data_path)

    # optionally use a subset of the test data
    if test_ratio < 1.0:
        N_use = max(1, int(len(X) * test_ratio))
        X, answers = X[:N_use], answers[:N_use]
    N_val = len(X)

    # distribute test cases
    per_rank = math.ceil(N_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, N_val)

    batch_size = 32
    num_batches = math.ceil((end - start) / batch_size)
    local_correct, local_total = 0, 0
    local_records = []

    tokenizer = get_tokenizer(tokenizer_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        for j in tqdm(range(num_batches), desc = "Evaluating"):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X = torch.from_numpy(X[s:e]).long().to(device)
            batch_answers = answers[s:e]
            # prompt_mask: True for prompt tokens, False for response (mask) tokens
            prompt_mask = (batch_X != mask_id)

            # also support the block diffusion training
            if cfg.training.strategy == "block":
                block_size = cfg.training.block_size
                samples_tensor = mdm_sampling_block(model, batch_X, block_size, mask_id, sampling, device)
            elif cfg.training.strategy == "arm":
                samples_tensor = arm_sampling(model, batch_X, mask_id, sampling, device)
            else:
                arm_init = cfg.model.get("arm_init", "none") != "none"
                samples_tensor = mdm_sampling(model, batch_X, mask_id, sampling, device, arm_init=arm_init, prompt_mask=prompt_mask)

            # decode prompts (tokens before the first mask position)
            batch_X_np = batch_X.cpu().numpy()
            prompt_texts = []
            for row in batch_X_np:
                mask_pos = np.where(row == mask_id)[0]
                prompt_end = int(mask_pos[0]) if len(mask_pos) > 0 else len(row)
                prompt_texts.append(tokenizer.decode(row[:prompt_end], skip_special_tokens=True))

            # tokenizer by default doesn't have mask_id; Qwen2 has no pad token so fall back to eos
            samples_tensor = samples_tensor.masked_fill(samples_tensor == mask_id, pad_id)

            # sample preproceessing, and extract the answer part
            sample_ids = samples_tensor.cpu().numpy()
            samples = tokenizer.batch_decode(sample_ids, skip_special_tokens=False)

            for prompt, sample, answer in zip(prompt_texts, samples, batch_answers):
                correct = evaluate_samples(sample, answer)
                if correct:
                    local_correct += 1
                local_total += 1
                local_records.append({"prompt": prompt, "response": sample, "gold": answer, "correct": correct})

    # accumulate success rates
    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()

    # # gather decoded responses to rank 0 and save
    # all_records = [None] * world_size
    # if world_size > 1 and dist.is_initialized():
    #     dist.all_gather_object(all_records, local_records)
    # else:
    #     all_records = [local_records]
    # if rank == 0:
    #     save_dir = os.path.join(logdir, "track", "gsm8k") if logdir else os.path.join("track", "gsm8k")
    #     os.makedirs(save_dir, exist_ok=True)
    #     fname = f"{metric_name}.jsonl" if metric_name else "results.jsonl"
    #     out_path = os.path.join(save_dir, fname)
    #     with open(out_path, "w") as f:
    #         for shard in all_records:
    #             for rec in shard:
    #                 f.write(json.dumps(rec) + "\n")

    return global_correct / global_total


_CALC_RE = re.compile(r"<<[^>]+=\s*([-+]?\d[\d,]*\.?\d*)\s*>>")

def evaluate_samples(sample: str, answer: str) -> bool:
    """
    Evaluate a standard GSM8K chain-of-thought response.
    Response format: reasoning with <<expr=N>> calculations, ending with '#### N'.
    Extraction priority:
      1. Last '#### N' marker
      2. Last '<<...=N>>' calculation result
    """
    pred_str = ""
    hash_match = _ANS_RE.search(sample)
    if hash_match:
        pred_str = hash_match.group(1).replace(",", "")
    else:
        calc_matches = _CALC_RE.findall(sample)
        if calc_matches:
            pred_str = calc_matches[-1].replace(",", "")

    pred = _to_number(pred_str)
    gold = _to_number(answer)
    return _numbers_equal(pred, gold)

# -----------------------------
# Code execution functions
# -----------------------------
class _Timeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _Timeout()


@contextlib.contextmanager
def _time_limit(timeout_s: float):
    """
    Hard wall-clock time limit using SIGALRM/ITIMER_REAL (POSIX).
    Note: works only in the main thread of the process.
    """
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    old_handler = None
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def _safe_exec_no_timer(code: str):
    """
    Executes code in a restricted environment (no timeout here).
    Timeout should be applied by wrapping the whole evaluate step with _time_limit().
    """
    import math as _math

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "enumerate": enumerate,
        "int": int, "float": float, "str": str, "bool": bool,
        "round": round,
        "print": print,
    }

    def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math":
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Import blocked: {name}")

    safe_builtins["__import__"] = _limited_import

    ns = {
        "__builtins__": safe_builtins,
        "math": _math,
    }

    # Reduce noisy compile-time warnings from weird generated code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(code, ns, ns)

    return ns

def _extract_code(text: str) -> str:
    """
    Heuristics:
      - If code fences exist, prefer fenced block.
      - Else, start from first 'def ' if present.
      - Strip special tokens.
      - Trim trailing garbage until it compiles (best-effort).
    """
    # Cut at common special tokens
    for stopper in ["<|endoftext|>", "<|eot_id|>", "</s>"]:
        if stopper in text:
            text = text.split(stopper, 1)[0]

    # Prefer fenced code
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    # If it contains 'def', slice from first def
    i = text.find("def ")
    if i != -1:
        text = text[i:]

    text = text.strip()

    # Best-effort trimming to make it syntactically valid
    # (Useful if sampling adds junk after valid Python)
    lines = text.splitlines()
    for k in range(0, min(50, len(lines))):
        candidate = "\n".join(lines[: len(lines) - k]).strip()
        if not candidate:
            continue
        try:
            compile(candidate, "<sample>", "exec")
            return candidate
        except SyntaxError:
            continue

    return text


# -----------------------------
# Numeric handling functions
# -----------------------------

def _numbers_equal(pred, gold):
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) or isinstance(gold, float):
        return abs(float(pred) - float(gold)) <= 1e-3
    return int(pred) == int(gold)

def _to_number(x):
    """
    Normalize return values to int or float where possible.
    """
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return None
        xf = float(x)
        if abs(xf - round(xf)) < 1e-6:
            return int(round(xf))
        return xf
    if isinstance(x, str):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", x)
        if not m:
            return None
        s = m.group(0).replace(",", "")
        if s.count(".") == 1:
            f = float(s)
            if abs(f - round(f)) < 1e-6:
                return int(round(f))
            return f
        return int(s)
    # tuples/lists etc -> not supported for GSM8K scoring
    return None


if __name__ == "__main__":
    # tokenize the GSM8K test set first
    # test_gsm8k_tokenization(MASK_ID)

    # sanity check the eval loop with one tinygsm example
    ds = load_dataset("TinyGSM/TinyGSM", split = "train")
    ex = ds[0]

    q, a  = ex["question"], ex["code"]

    ns = _safe_exec_no_timer( _extract_code(q + "\n" + a) )
    out = ns["simple_math_problem"]()
    gold = str(_to_number(out))

    ok = evaluate_samples( a , gold)

    print("Sanity check passed: ", ok)
    print("--------------------------------")
    print("Question: ", q)
    print("--------------------------------")
    print("Code: ", a)
    print("--------------------------------")
    print("Answer: ", gold)