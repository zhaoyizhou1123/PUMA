import json
import os

def pad_sequence(dataitem, max_prompt_length):
    seq = dataitem["sequence"]
    prompt_length = dataitem["prompt_length"]
    if prompt_length > max_prompt_length:
        raise ValueError(f"Prompt length {prompt_length} exceeds max prompt length {max_prompt_length}")
    assert (max_prompt_length - prompt_length) % 2 == 0
    padding_seq = [1] * (max_prompt_length - prompt_length) 
    dataitem["sequence"] = padding_seq + seq
    dataitem["prompt_length"] = max_prompt_length
    return dataitem

if __name__ == "__main__":
    data_dir = "data/maze17x17_dfs"
    max_prompt_length = 256
    # rename train.jsonl to train_original.jsonl
    os.rename(f"{data_dir}/train.jsonl", f"{data_dir}/train_original.jsonl")
    with open(f"{data_dir}/train_original.jsonl", "r") as fr, open(f"{data_dir}/train.jsonl", "w") as fw:
        for line in fr:
            dataitem = json.loads(line)
            padded_dataitem = pad_sequence(dataitem, max_prompt_length)
            fw.write(json.dumps(padded_dataitem) + "\n")

    os.rename(f"{data_dir}/test.jsonl", f"{data_dir}/test_original.jsonl")
    with open(f"{data_dir}/test_original.jsonl", "r") as fr, open(f"{data_dir}/test.jsonl", "w") as fw:
        for line in fr:
            dataitem = json.loads(line)
            padded_dataitem = pad_sequence(dataitem, max_prompt_length)
            fw.write(json.dumps(padded_dataitem) + "\n")
