# Validation cost against depth for the kink-free trial network: one dot per seed and a line through the means,
# so both the trend (the mean levels off) and the spread between seeds (it grows with depth) can be seen.
import numpy as np
import matplotlib.pyplot as plt

from src.training import RESULTS_DIR
from src.plotting import save_fig

scan = np.load(RESULTS_DIR / "depth_scan_kink_free.npz")
layers, validation, train_time = scan["layers"], scan["validation"], scan["train_time"]   # layers x seeds
mean, std = validation.mean(axis=1), validation.std(axis=1)
chosen = 5                                                     # the depth used in the rest of part d

for L, m, s, v, t in zip(layers, mean, std, validation, train_time.mean(axis=1)):
    print(f"{L} layers: validation {m:.2e} ± {s:.0e} (max/min over seeds {v.max() / v.min():.1f}), training {t:.0f} s")

fig, ax = plt.subplots()
for i in range(validation.shape[1]):                           # the seeds of each depth in a vertical column
    ax.plot(layers, validation[:, i], "o", color="tab:green", alpha=0.45, ms=4,
            label="one seed" if i == 0 else None)
ax.plot(layers, mean, "-", color="tab:green", lw=1.5, label="mean over seeds")
ax.axvline(chosen, color="0.8", lw=6, zorder=0)
ax.set_yscale("log")
ax.set_xticks(layers)
ax.set_xlabel("hidden layers (50 nodes each)")
ax.set_ylabel("validation cost\n(mean squared residual)")
ax.legend(fontsize=7)
save_fig(fig, "depth")
