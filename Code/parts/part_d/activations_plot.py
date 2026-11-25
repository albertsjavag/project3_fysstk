# Bar chart of the activation scan: validation cost for each activation, mean over seeds with a band of ± one
# standard deviation. ReLU is hatched: its low validation cost cannot be trusted, because autodiff does not see
# the kinks of a ReLU network (the second derivative of a kink is a delta, which autodiff returns as zero).
import numpy as np
import matplotlib.pyplot as plt

from src.training import RESULTS_DIR
from src.plotting import save_fig

methods = {"kink_free": ("kink-free trial", "tab:green"), "pinn": ("PINN", "tab:blue")}
finished = [m for m in methods if (RESULTS_DIR / f"activation_scan_{m}.npz").exists()]
width = 0.8 / len(finished)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
lowest = np.inf
for k, method in enumerate(finished):
    label, color = methods[method]
    scan = np.load(RESULTS_DIR / f"activation_scan_{method}.npz")
    activations = list(scan["activations"])
    mean, std = scan["validation"].mean(axis=1), scan["validation"].std(axis=1)
    time = scan["train_time"].mean(axis=1)
    for j, activation in enumerate(activations):
        print(f"{label:16s} {activation:9s} validation {mean[j]:.2e} ± {std[j]:.0e}, training {time[j]:.1f} s")

    positions = np.arange(len(activations)) + (k - (len(finished) - 1) / 2) * width
    low = np.maximum(mean - std, 1e-12)                      # keep the bar above zero on the log axis
    lowest = min(lowest, low.min())
    bars = ax.bar(positions, low, width, color=color, label=label)                       # solid up to mean - std
    ax.bar(positions, mean + std - low, width, bottom=low, color=color, alpha=0.35)      # band mean ± std
    ax.hlines(mean, positions - width / 2, positions + width / 2, color=color, linewidth=1.5)  # the mean itself
    relu = activations.index("relu")
    bars[relu].set_facecolor("white")
    bars[relu].set_edgecolor(color)
    bars[relu].set_hatch("////")

ax.set_yscale("log")
ax.set_ylim(bottom=lowest / 2)                               # every solid part visible
ax.set_xticks(np.arange(len(activations)), activations, rotation=30, ha="right")
ax.set_ylabel("validation cost\n(mean squared residual)")
ax.legend(fontsize=7, loc="upper left")
save_fig(fig, "activations")
