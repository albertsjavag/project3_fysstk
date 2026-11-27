import json

from src.training import train_config, RESULTS_DIR

n_seeds = 10

# The settings chosen in the scans of part d (lowest mean validation cost; ReLU ruled out by theory):
#   architecture and learning rate: outputs/parts.part_d.architecture_plot.txt
#   activation:                     outputs/parts.part_d.activations_plot.txt
#   collocation points:             outputs/parts.part_d.collocation_plot.txt
#   PINN weight w:                  outputs/parts.part_d.loss_weights_plot.txt
kink_free = {"sizes": [2, 50, 50, 50, 50, 50, 1], "learning_rate": 1e-2, "activation": "gelu",
             "n_col": 4000, "placement": "kink"}
pinn = {"sizes": [2, 50, 50, 50, 1], "learning_rate": 3e-3, "activation": "gelu",
        "n_col": 4000, "placement": "uniform", "w_initial": 10.0, "w_boundary": 10.0}

# name: (method, settings). The kink-free net is also trained with uniform points: the scan picked the kink
# placement by a small margin, and only the exact solution can tell whether it helps near the kink. The two
# controls use the kink-free settings: the naive trial network, and the kink-free network with ReLU. Both have
# a low validation cost but hidden kinks.
configs = {
    "kink_free": ("kink_free", kink_free),
    "kink_free_uniform": ("kink_free", {**kink_free, "placement": "uniform"}),     # the same, but uniform points
    "pinn": ("pinn", pinn),
    "naive": ("naive", kink_free),
    "relu": ("kink_free", {**kink_free, "activation": "relu"}),     # the same settings, but ReLU
}

with open(RESULTS_DIR / "final_configs.json", "w") as f:       # final_test.py reads the settings from here
    json.dump({name: {"method": method, **settings} for name, (method, settings) in configs.items()}, f, indent=1)

for name, (method, settings) in configs.items():
    for i in range(n_seeds):
        train_config(method, seed_offset=i, **settings, name=f"final_{name}_seed{i}")
