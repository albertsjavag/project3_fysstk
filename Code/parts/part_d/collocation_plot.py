# The collocation study: prints the scan saved by collocation.py, and draws where the points are placed with
# the kink placement (kink_points in src/training.py): half uniform, half close to the kink.
import numpy as np
import matplotlib.pyplot as plt

from src.problems import SEED, L
from src.training import RESULTS_DIR, kink_points
from src.plotting import save_fig

# --- The scan: validation cost for each number of points and placement ---

labels = {"kink_free": "kink-free trial", "pinn": "PINN"}
for method, label in labels.items():
    scan = np.load(RESULTS_DIR / f"collocation_scan_{method}.npz")
    mean, std = scan["validation"].mean(axis=-1), scan["validation"].std(axis=-1)     # n_cols x placements
    best = np.unravel_index(np.argmin(mean), mean.shape)
    for j, n_col in enumerate(scan["n_cols"]):
        for p, placement in enumerate(scan["placements"]):
            mark = "  <- best" if (j, p) == best else ""
            print(f"{label:16s} {n_col:5d} {placement:8s} points: validation {mean[j, p]:.2e} ± {std[j, p]:.0e}, "
                  f"training {scan['train_time'][j, p].mean():.0f} s{mark}")

# --- The figure ---

n = 1000                                       # the default number of points, so the dots stay visible
points = np.asarray(kink_points(n, -L, L, np.random.default_rng(SEED + 100)))

fig, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], ".", ms=1.5, color="tab:blue")
ax.axvline(0, color="0.6", ls="--", lw=0.8)                                              # the strike
ax.set_xlabel(r"$x$ (0 = strike)")
ax.set_ylabel(r"$s = \tau / \tau_{max}$")
save_fig(fig, "collocation")
