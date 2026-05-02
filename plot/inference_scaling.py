"""
Plot inference scaling: accuracy vs NFE for three methods.

NFE definitions:
  gibbs_edit / gibbs_standard : NFE = 81 * (1 + edit_step)
  ReMDM                       : NFE = num_steps
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_PROGRESSIVE = (
    "/home/frankwu2/mdm_correction/logs/eval_progressive_edit_gibbs_edit_460k_7098388.log"
)
LOG_STANDARD = (
    "/home/frankwu2/mdm_correction/logs/eval_standard_500k_gibbs_standard_7097769.log"
)
LOG_REMDM = (
    "/home/frankwu2/PUMA/logs/eval_remdm_nfe_sweep_7247217.log"
)
LOG_PROSECO = (
    "/home/frankwu2/mdm_correction/logs/eval_proseco_gibbs_edit.log"
)


def parse_gibbs(path, key_prefix):
    """Return list of (nfe, acc) from a gibbs edit_step sweep log."""
    pattern = re.compile(
        rf"Validation Accuracy {key_prefix}.*?editstep_(\d+):\s+([\d.]+)"
    )
    results = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                edit_step = int(m.group(1))
                acc = float(m.group(2))
                nfe = 81 * (1 + edit_step)
                results.append((nfe, acc))
    results.sort()
    return results


def parse_remdm(path):
    """Return list of (nfe, acc) from a ReMDM num_steps sweep log."""
    results = []
    current_steps = None
    step_pat = re.compile(r"--- num_steps=(\d+) ---")
    acc_pat = re.compile(r"variant=rescale\s+eta=[\d.]+\s+->\s+acc=([\d.]+)")
    with open(path) as f:
        for line in f:
            m = step_pat.search(line)
            if m:
                current_steps = int(m.group(1))
                continue
            m = acc_pat.search(line)
            if m and current_steps is not None:
                acc = float(m.group(1))
                results.append((current_steps, acc))
                current_steps = None
    results.sort()
    return results


gibbs_edit = parse_gibbs(LOG_PROGRESSIVE, "gibbs_edit")
gibbs_standard = parse_gibbs(LOG_STANDARD, "gibbs_standard")
remdm = parse_remdm(LOG_REMDM)
proseco = parse_gibbs(LOG_PROSECO, "gibbs_edit")

fig, ax = plt.subplots(figsize=(6, 4))

for data, label, color, marker in [
    (gibbs_edit,     "Gibbs-Edit (progressive)",  "tab:blue",   "o"),
    (gibbs_standard, "Gibbs-Standard (standard)", "tab:orange", "s"),
    (remdm,          "ReMDM (standard)",           "tab:green",  "^"),
    (proseco,        "Gibbs-Edit (proseco)",        "tab:red",    "D"),
]:
    nfes = [nfe for nfe, _ in data]
    accs = [acc for _, acc in data]
    ax.plot(nfes, accs, marker=marker, label=label, color=color, linewidth=1.5, markersize=5)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("NFE (log scale)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Inference Scaling on Hard Sudoku", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 1)

fig.tight_layout()
out_base = "/home/frankwu2/PUMA/plot/inference_scaling"
fig.savefig(f"{out_base}.pdf")
fig.savefig(f"{out_base}.png", dpi=150)
print(f"Saved {out_base}.pdf and {out_base}.png")
