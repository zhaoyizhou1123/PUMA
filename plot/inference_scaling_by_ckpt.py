"""
3 plots — one per pretraining checkpoint (progressive_edit, standard, proseco).
Each plot shows all 4 inference methods (gibbs_edit, gibbs_standard, proseco, remdm).

NFE = 81 * (1 + edit_step), edit_step in {0,1,2,4,8,16,32,64,128,256}
ReMDM num_steps are matched to the same NFE values.
"""

import re
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------
LOGS = {
    "progressive_edit": {
        "gibbs_edit":     "/home/frankwu2/mdm_correction/logs/eval_progressive_edit_460k_7566917.log",
        "gibbs_standard": "/home/frankwu2/mdm_correction/logs/eval_progressive_edit_460k_gibbs_standard_7566918.log",
        "proseco":        "/home/frankwu2/mdm_correction/logs/eval_progressive_edit_460k_proseco_7643495.log",
        "remdm":          "/home/frankwu2/PUMA/logs/eval_remdm_nfe_sweep_progressive_7566893.log",
    },
    "standard": {
        "gibbs_edit":     "/home/frankwu2/mdm_correction/logs/eval_standard_500k_gibbs_edit_7566900.log",
        "gibbs_standard": "/home/frankwu2/mdm_correction/logs/eval_standard_500k_7566899.log",
        "proseco":        "/home/frankwu2/mdm_correction/logs/eval_standard_500k_proseco_7643497.log",
        "remdm":          "/home/frankwu2/PUMA/logs/eval_remdm_nfe_sweep_7566896.log",
    },
    "proseco": {
        "gibbs_edit":     "/home/frankwu2/mdm_correction/logs/eval_proseco_500k_gibbs_edit_2211230.log",
        "gibbs_standard": "/home/frankwu2/mdm_correction/logs/eval_proseco_500k_gibbs_standard_2211231.log",
        "proseco":        "/home/frankwu2/mdm_correction/logs/eval_proseco_500k_proseco_2227557.log",
        "remdm":          "/home/frankwu2/PUMA/logs/eval_remdm_nfe_sweep_proseco_2212437.log",
    },
}

TITLES = {
    "progressive_edit": "Progressive-Edit Checkpoint",
    "standard":         "Standard Checkpoint",
    "proseco":          "Proseco Checkpoint",
}

# Canonical NFE values (edit_steps 0,1,2,4,8,16,32,64,128,256 → 81*(1+k))
NFE_VALUES = [81 * (1 + k) for k in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]]
NFE_TO_IDX = {nfe: i for i, nfe in enumerate(NFE_VALUES)}
TICK_LABELS = [str(n) for n in NFE_VALUES]

STYLES = {
    "gibbs_edit":     ("Gibbs-Edit",     "tab:blue",   "o", "-"),
    "gibbs_standard": ("Gibbs-Standard", "tab:orange", "s", "-"),
    "proseco":        ("Proseco",         "tab:red",    "D", "-"),
    "remdm":          ("ReMDM",           "tab:green",  "^", "-"),
}

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_gibbs(path, key_prefix):
    pattern = re.compile(
        rf"Validation Accuracy {key_prefix}.*?editstep_(\d+):\s+([\d.]+)"
    )
    nfe_to_acc = {}
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                nfe = 81 * (1 + int(m.group(1)))
                nfe_to_acc[nfe] = float(m.group(2))
    return nfe_to_acc


def parse_remdm(path):
    nfe_to_acc = {}
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
                nfe_to_acc[current_steps] = float(m.group(1))
                current_steps = None
    return nfe_to_acc

# ---------------------------------------------------------------------------
# One plot per checkpoint
# ---------------------------------------------------------------------------

for ckpt_name, logs in LOGS.items():
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, (label, color, marker, ls) in STYLES.items():
        path = logs[method]
        if method == "remdm":
            nfe_to_acc = parse_remdm(path)
        else:
            nfe_to_acc = parse_gibbs(path, method)

        xs, ys = [], []
        for nfe in NFE_VALUES:
            if nfe in nfe_to_acc:
                xs.append(NFE_TO_IDX[nfe])
                ys.append(nfe_to_acc[nfe])

        ax.plot(xs, ys, marker=marker, label=label, color=color,
                linewidth=1.5, markersize=5, linestyle=ls)

    ax.set_xticks(range(len(NFE_VALUES)))
    ax.set_xticklabels(TICK_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("NFE", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(TITLES[ckpt_name], fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out_base = f"/home/frankwu2/PUMA/plot/inference_scaling_{ckpt_name}"
    fig.savefig(f"{out_base}.pdf")
    fig.savefig(f"{out_base}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_base}.pdf/.png")
