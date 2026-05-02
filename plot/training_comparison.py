"""
Scatter plot: accuracy vs NFE for each training method at its default eval setting.

NFE values (manually computed from sampling code):
  standard        : 81   (no editing, 81 unmasking steps)
  puma            : 81   (no editing, 81 unmasking steps)
  remdm           : 81   (num_steps=81)
  remedi          : 81   (1 pass/step, unmasking_num=1)
  prism           : 81   (1 pass/step, unmasking_num=1)
  backplay        : ~101 (81 base + ~20 correction passes at stride=4)
  proseco         : 189  (edit_freq=3, edit_step=4 -> 54*1 + 27*5)
  progressive_edit: 189  (edit_freq=3, edit_step=4 -> 54*1 + 27*5)
"""

import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------
LOGS = {
    "progressive_edit": "/home/frankwu2/PUMA/logs/train_sudoku_edit_hard_6887018.log",
    "proseco":          "/home/frankwu2/PUMA/logs/proseco.log",
    "puma":             "/home/frankwu2/PUMA/logs/puma.log",
    "standard":         "/home/frankwu2/PUMA/logs/train_sudoku_standard_hard_6893656.log",
    "remedi":           "/home/frankwu2/PUMA/logs/train_sudoku_remedi_hard_7010114.log",
    "remdm":            "/home/frankwu2/PUMA/logs/eval_remdm_nfe_sweep_7247217.log",
    "backplay":         "/home/frankwu2/PUMA/logs/train_sudoku_backplay_hard_7008718.log",
    "prism":            "/home/frankwu2/PUMA/logs/train_sudoku_prism_hard_7021953.log",
}

# NFE at default eval setting (see module docstring)
NFE = {
    "standard":         81,
    "puma":             81,
    "remdm":            81,
    "remedi":           81,
    "prism":            81,
    "backplay":         101,
    "proseco":          189,
    "progressive_edit": 189,
}

# Display names
LABELS = {
    "standard":         "Standard MDM",
    "puma":             "PUMA",
    "remedi":           "RemeDi",
    "prism":            "PRISM",
    "backplay":         "BackPlay",
    "proseco":          "ProSeCo",
    "progressive_edit": "Progressive Edit (Ours)",
}

SKIP = {"remdm"}

# ---------------------------------------------------------------------------
# Parsers — extract the final (last) reported accuracy from each log
# ---------------------------------------------------------------------------

def last_match(path, pattern):
    """Return the last regex match float in the file, or None."""
    result = None
    with open(path) as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                result = float(m.group(1))
    return result


def parse_accuracy(name, path):
    if name == "progressive_edit":
        # "Validation Accuracy top_k_unmasking_1_editfreq_3_editstep_4: 0.598"
        return last_match(path, r"Validation Accuracy top_k_unmasking_1_editfreq_3_editstep_4:\s+([\d.]+)")

    if name == "proseco":
        return last_match(path, r"Validation Accuracy top_k_unmasking_1_editfreq_3_editstep_4:\s+([\d.]+)")

    if name in ("puma", "standard"):
        # "Validation Accuracy top_k_unmasking_1: 0.508"  (must NOT match editfreq variant)
        return last_match(path, r"Validation Accuracy top_k_unmasking_1:\s+([\d.]+)")

    if name == "remedi":
        # "Step 50000  val_loss=...  std_acc=...  remedi_acc=0.5020"
        return last_match(path, r"remedi_acc=([\d.]+)")

    if name == "backplay":
        # "Step 50000  ...  backplay_acc=0.5190  ..."
        return last_match(path, r"backplay_acc=([\d.]+)")

    if name == "prism":
        # "Step 50000  val_loss=...  std_acc=...  prism_acc=0.5080"
        return last_match(path, r"prism_acc=([\d.]+)")

    if name == "remdm":
        # Find the block after "--- num_steps=81 ---" and grab the acc
        in_block = False
        with open(path) as f:
            for line in f:
                if re.search(r"---\s*num_steps=81\s*---", line):
                    in_block = True
                    continue
                if in_block:
                    m = re.search(r"variant=\w+.*?acc=([\d.]+)", line)
                    if m:
                        return float(m.group(1))
                    if re.search(r"---\s*num_steps=", line):
                        break  # moved to next block
        return None

    return None


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
data = {}
for name, path in LOGS.items():
    acc = parse_accuracy(name, path)
    if acc is None:
        print(f"WARNING: could not parse accuracy for {name}")
    data[name] = acc

data["prism"] = 0.5220

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

# Colour groups: pretrain-only vs fine-tuning vs inference-only
COLOR = {
    "standard":         "#4878CF",  # blue
    "puma":             "#6ACC65",  # green
    "proseco":          "#D65F5F",  # red
    "progressive_edit": "#B47CC7",  # purple
    "remdm":            "#C4AD66",  # tan  (inference-only)
    "remedi":           "#77BEDB",  # sky blue
    "prism":            "#F28E2B",  # orange
    "backplay":         "#59A14F",  # dark green
}

MARKER = {
    "standard":         "o",
    "puma":             "o",
    "proseco":          "s",
    "progressive_edit": "s",
    "remdm":            "^",
    "remedi":           "D",
    "prism":            "P",
    "backplay":         "X",
}

for name, acc in data.items():
    if acc is None or name in SKIP:
        continue
    nfe = NFE[name]
    is_ours = (name == "progressive_edit")
    ax.scatter(
        nfe, acc,
        color=COLOR[name],
        marker=MARKER[name],
        s=130 if is_ours else 100,
        zorder=3,
        label=LABELS[name],
    )
    ax.annotate(
        LABELS[name],
        xy=(nfe, acc),
        xytext=(-8, 0),
        textcoords="offset points",
        fontsize=11 if is_ours else 9,
        fontweight="bold" if is_ours else "normal",
        va="center",
        ha="right",
        color=COLOR[name],
    )

ax.set_xlabel("NFE (Number of Function Evaluations)", fontsize=11)
ax.set_ylabel("Puzzle Accuracy", fontsize=11)
ax.set_title("Training Methods — Default Eval Setting\n(Sudoku Hard, 500k steps)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(left=0)

# Zoom to the range of non-skipped methods
plotted_accs = [acc for name, acc in data.items() if name not in SKIP and acc is not None]
ymin = min(plotted_accs) - 0.02
ymax = max(plotted_accs) + 0.02
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("/home/frankwu2/PUMA/plot/training_comparison.png", dpi=150)
plt.savefig("/home/frankwu2/PUMA/plot/training_comparison.pdf")
print("Saved training_comparison.png / .pdf")

# Print parsed values for verification
print("\nParsed accuracies:")
for name in LABELS:
    print(f"  {LABELS[name]:26s}  NFE={NFE[name]:4d}  acc={data.get(name)}")
