import re
import matplotlib.pyplot as plt

pattern = r"Validation Accuracy top_k_unmasking_1_editfreq_1_editstep_(\d+): ([0-9.]+)"

data = []
with open("../eval_standard.log") as f:
    for line in f:
        m = re.search(pattern, line)
        if m:
            data.append((int(m.group(1)), float(m.group(2))))
data.sort()
steps, accs = zip(*data)

plt.figure(figsize=(8, 5))
plt.plot(steps, accs, marker='o', label="gibbs standard")
plt.xlabel("Edit Step")
plt.ylabel("Accuracy")
plt.title("Standard Accuracy vs Edit Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("standard.png", dpi=150)
print("Saved stats/standard.png")
