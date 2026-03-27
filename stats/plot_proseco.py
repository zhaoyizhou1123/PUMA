import re
import matplotlib.pyplot as plt

pattern = r"Validation Accuracy top_k_unmasking_1_editfreq_1_editstep_(\d+): ([0-9.]+)"

sources = [
    ("eval_proseco.log",       "Proseco (proseco correction)"),
    ("eval_proseco_gibbs.log", "Proseco (gibbs correction)"),
]

plt.figure(figsize=(8, 5))
for fname, label in sources:
    data = []
    with open(f"../{fname}") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                data.append((int(m.group(1)), float(m.group(2))))
    data.sort()
    steps, accs = zip(*data)
    plt.plot(steps, accs, marker='o', label=label)

plt.xlabel("Edit Step")
plt.ylabel("Accuracy")
plt.title("Proseco Accuracy vs Edit Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("proseco.png", dpi=150)
print("Saved stats/proseco.png")
