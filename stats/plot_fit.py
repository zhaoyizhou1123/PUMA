import re
import numpy as np
from scipy.optimize import curve_fit
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
steps = np.array(steps, dtype=float)
accs  = np.array(accs,  dtype=float)

MODELS = [
    dict(
        name="Stretched exp",
        fn=lambda n, A, B, k, beta: A + B * np.exp(-k * (n + 1) ** beta),
        p0=lambda: [accs.max(), accs.min() - accs.max(), 0.01, 0.5],
        label=lambda p, r2: (rf"Stretched exp: $A + B\,e^{{-k\,(n+1)^\beta}}$"
                             f"\n$A={p[0]:.4f}$, $B={p[1]:.4f}$, $k={p[2]:.5f}$, $\\beta={p[3]:.3f}$, $R^2={r2:.4f}$"),
    ),
]

# Weight proportional to (n+1) so later points dominate the fit
sigma = 1.0 / (steps + 1)

n_dense = np.linspace(0, steps.max() * 1.1, 500)
colors = ["tomato", "seagreen", "darkorange", "mediumpurple"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(steps, accs, color="steelblue", zorder=5, label="Observed")

for model, color in zip(MODELS, colors):
    try:
        popt, _ = curve_fit(model["fn"], steps, accs, p0=model["p0"](),
                            sigma=sigma, absolute_sigma=True, maxfev=20000)
        pred = model["fn"](steps, *popt)
        r2 = 1 - np.sum((accs - pred) ** 2) / np.sum((accs - accs.mean()) ** 2)
        ax.plot(n_dense, model["fn"](n_dense, *popt), color=color, linewidth=2,
                label=model["label"](popt, r2))
    except Exception as e:
        print(f"{model['name']} fit failed: {e}")

ax.set_xlabel("Edit Step")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Edit Step — Curve Fits (Standard)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("standard_fit.png", dpi=150)
print("Saved stats/standard_fit.png")
