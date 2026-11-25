
import numpy as np
from src.problems import SEED, L, tau_max, bs_initial, bs_left_boundary, bs_right_boundary
from src.costs import pinn_terms
from src.training import train_config, RESULTS_DIR, interior_points, initial_points, boundary_points

weights = [1.0, 10.0, 100.0, 1000.0, 10000.0]
n_seeds = 3

arch = np.load(RESULTS_DIR / "architecture_scan_pinn.npz")
a, b, c = np.unravel_index(np.argmin(arch["validation"].mean(axis=-1)), arch["validation"].shape[:3])
sizes = (2,) + (int(arch["nodes"][b]),) * int(arch["layers"][a]) + (1,)
learning_rate = float(arch["learning_rates"][c])
act = np.load(RESULTS_DIR / "activation_scan_pinn.npz")
mean = act["validation"].mean(axis=1)
mean[list(act["activations"]).index("relu")] = np.inf     # ReLU is ruled out by theory, not by its cost
activation = str(act["activations"][np.argmin(mean)])

X_val = interior_points(500, -L, L, np.random.default_rng(SEED + 200))
X_initial = initial_points(200, -L, L)
X_left, X_right = boundary_points(100, -L), boundary_points(100, L)

terms = np.zeros((len(weights), n_seeds, 3))       # residual, initial, boundary
train_time = np.zeros((len(weights), n_seeds))
for j, w in enumerate(weights):
    for i in range(n_seeds):
        run = train_config("pinn", seed_offset=i, sizes=sizes, learning_rate=learning_rate,
                           activation=activation, w_initial=w, w_boundary=w)
        terms[j, i] = pinn_terms(run["params"], X_val, X_initial, X_left, X_right, bs_initial,
                                 bs_left_boundary, bs_right_boundary, tau_max, activation)
        train_time[j, i] = run["timing"]["train"]

np.savez(RESULTS_DIR / "loss_weight_scan.npz", terms=terms, train_time=train_time, weights=weights,
         sizes=sizes, learning_rate=learning_rate, activation=activation)
