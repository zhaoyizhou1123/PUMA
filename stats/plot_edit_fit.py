import re
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

pattern = r"Validation Accuracy top_k_unmasking_1_editfreq_1_editstep_(\d+): ([0-9.]+)"

data = []
with open("../eval.log") as f:
    for line in f:
        m = re.search(pattern, line)
        if m:
            data.append((int(m.group(1)), float(m.group(2))))
data.sort()
steps, accs = zip(*data)
steps = np.array(steps, dtype=float)
accs  = np.array(accs,  dtype=float)

acc0 = accs[0]  # baseline at step 0

def exp_model(n, A, k):
    return A - (A - acc0) * np.exp(-k * n)

(A, k), _ = curve_fit(exp_model, steps, accs, p0=[0.9, 0.005], maxfev=10000)
pred = exp_model(steps, A, k)
r2 = 1 - np.sum((accs - pred)**2) / np.sum((accs - accs.mean())**2)

n_dense = np.linspace(0, 400, 500)
fit_dense = exp_model(n_dense, A, k)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(steps, accs, color="steelblue", zorder=5, label="Observed")
ax.plot(n_dense, fit_dense, color="tomato", linewidth=2,
        label=rf"Fit: $A - (A-{acc0:.3f})\,e^{{-kn}}$"
              f"\n$A={A:.4f}$, $k={k:.5f}$, $R^2={r2:.4f}")

ax.set_xlabel("Edit Step")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Edit Step — Exponential Fit")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("edit_fit.png", dpi=150)
print("Saved stats/edit_fit.png")
