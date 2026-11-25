
import numpy as np
import matplotlib.pyplot as plt

from src.training import RESULTS_DIR
from src.plotting import save_fig

scan = np.load(RESULTS_DIR / "loss_weight_scan.npz")
weights = scan["weights"]
terms = scan["terms"]                 # weights x seeds x 3 (residual, initial, boundary)
train_time = scan["train_time"]       # weights x seeds

total = terms.sum(axis=2).mean(axis=1)
best = np.argmin(total)

names = ["residual", "initial", "boundary"]
for j, w in enumerate(weights):
    parts = ", ".join(f"{n} {terms[j, :, k].mean():.1e} ± {terms[j, :, k].std():.0e}" for k, n in enumerate(names))
    mark = "  <- best" if j == best else ""
    print(f"w = {w:7g}: {parts}, total {total[j]:.2e}, training {train_time[j].mean():.0f} s{mark}")

fig, ax = plt.subplots()
for k, (name, color) in enumerate([("residual (equation)", "tab:blue"), ("initial condition (payoff)", "tab:red")]):
    mean, std = terms[:, :, k].mean(axis=1), terms[:, :, k].std(axis=1)
    ax.plot(weights, mean, "o-", color=color, label=name)
    ax.fill_between(weights, np.maximum(mean - std, 1e-16), mean + std, color=color, alpha=0.3)
ax.axvline(weights[best], ls="--", color="0.5")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"weight $w$ on the conditions")
ax.set_ylabel("validation term (unweighted)")
ax.legend(fontsize=7)
save_fig(fig, "loss_weights")