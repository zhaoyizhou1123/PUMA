"""
Plot comparing gibbs_edit and gibbs_standard with and without low-confidence masking.

NFE = 81 * (1 + edit_step)
"""

import re
import matplotlib.pyplot as plt

LOG_EDIT         = "/home/frankwu2/mdm_correction/logs/eval_progressive_edit_460k_7098388.log"
LOG_EDIT_LOW     = "/home/frankwu2/mdm_correction/logs/eval_progressive_edit_460k_low_conf_7213528.log"
LOG_STANDARD     = "/home/frankwu2/mdm_correction/logs/eval_standard_500k_7097769.log"
LOG_STANDARD_LOW = "/home/frankwu2/mdm_correction/logs/eval_standard_500k_low_conf_7213524.log"


def parse_gibbs(path, key_prefix):
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


gibbs_edit         = parse_gibbs(LOG_EDIT,         "gibbs_edit")
gibbs_edit_low     = parse_gibbs(LOG_EDIT_LOW,     "gibbs_edit")
gibbs_standard     = parse_gibbs(LOG_STANDARD,     "gibbs_standard")
gibbs_standard_low = parse_gibbs(LOG_STANDARD_LOW, "gibbs_standard")

nfes = [nfe for nfe, _ in gibbs_edit]
xs = list(range(len(nfes)))
tick_labels = [f"{n:,}" for n in nfes]

fig, ax = plt.subplots(figsize=(6, 4))

for data, label, color, marker, ls in [
    (gibbs_edit,         "Gibbs-Edit (progressive)",                "tab:blue",   "o", "-"),
    (gibbs_edit_low,     "Gibbs-Edit (progressive) + low-conf",     "tab:blue",   "o", "--"),
    (gibbs_standard,     "Gibbs-Standard (standard)",               "tab:orange", "s", "-"),
    (gibbs_standard_low, "Gibbs-Standard (standard) + low-conf",    "tab:orange", "s", "--"),
]:
    accs = [acc for _, acc in data]
    ax.plot(xs, accs, marker=marker, label=label, color=color,
            linewidth=1.5, markersize=5, linestyle=ls)

ax.set_xticks(xs)
ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
ax.set_xlabel("NFE", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Low-Confidence Masking Comparison", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 1)

fig.tight_layout()
out_base = "/home/frankwu2/PUMA/plot/low_confidence_comparison"
fig.savefig(f"{out_base}.pdf")
fig.savefig(f"{out_base}.png", dpi=150)
print(f"Saved {out_base}.pdf and {out_base}.png")
