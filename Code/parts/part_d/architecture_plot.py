# 3D scatter of the architecture scan, adapted from the example report's grid-search figure (D. Haas,
# FYS-STK4155 project 3, GridSearch.py). Reads what architecture.py saved; draws one panel per finished scan.

import numpy as np
import matplotlib.pyplot as plt

from src.network import create_layers, n_parameters
from src.training import RESULTS_DIR
from src.plotting import save_fig, PAGE

labels = {"kink_free": "kink-free trial", "pinn": "PINN"}
finished = [m for m in labels if (RESULTS_DIR / f"architecture_scan_{m}.npz").exists()]

fig = plt.figure(figsize=(PAGE[0], 3.3), layout="none")
for k, method in enumerate(finished):
    stored = np.load(RESULTS_DIR / f"architecture_scan_{method}.npz")
    layers, nodes, learning_rates = stored["layers"], stored["nodes"], stored["learning_rates"]
    mean = stored["validation"].mean(axis=-1)            # mean over seeds
    std = stored["validation"].std(axis=-1)
    time = stored["train_time"].mean(axis=-1)

    # Best configuration: lowest mean validation cost.
    a, b, c = np.unravel_index(np.argmin(mean), mean.shape)
    sizes = [2] + [int(nodes[b])] * int(layers[a]) + [1]
    print(f"{labels[method]}: best {layers[a]} layers x {nodes[b]} nodes, learning rate {learning_rates[c]:g}: "
          f"validation {mean[a, b, c]:.2e} ± {std[a, b, c]:.0e}, training {time[a, b, c]:.1f} s, "
          f"{n_parameters(create_layers(sizes, np.random.default_rng(0)))} parameters")

    # One dot per configuration, coloured by log10 of the mean validation cost; the best one ringed in red.
    ax = fig.add_subplot(1, len(finished), k + 1, projection="3d")
    L, N, R = np.meshgrid(layers, nodes, np.log10(learning_rates), indexing="ij")
    # depthshade=False: otherwise points further back are faded, and the colour no longer shows only the cost.
    dots = ax.scatter(L.ravel(), N.ravel(), R.ravel(), c=np.log10(mean).ravel(), cmap="magma", s=40, depthshade=False)
    ax.scatter(layers[a], nodes[b], np.log10(learning_rates[c]), s=110, facecolors="none", edgecolors="red")
    ax.set_xlabel("layers", labelpad=-2)
    ax.set_ylabel("nodes per layer", labelpad=-2)
    ax.set_zlabel(r"$\log_{10}$ learning rate", labelpad=-2)
    ax.set_xticks(layers)
    ax.set_yticks(nodes)
    ax.set_zticks(np.log10(learning_rates))
    ax.set_zticklabels([f"{v:.1f}" for v in np.log10(learning_rates)])
    ax.tick_params(pad=0)
    ax.text2D(0.02, 0.95, labels[method], transform=ax.transAxes)
    colorbar = fig.colorbar(dots, ax=ax, location="left", pad=0.02, shrink=0.6)
    colorbar.set_label(r"$\log_{10}$ validation cost (mean squared residual)", fontsize=7)

fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.98, wspace=0.45)
with plt.rc_context({"savefig.bbox": "standard"}):       # the tight box cuts 3D axis labels
    save_fig(fig, "architecture")
