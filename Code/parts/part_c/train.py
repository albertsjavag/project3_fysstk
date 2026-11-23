import numpy as np
import jax

from src.problems import SEED, L, tau_max, bs_initial, bs_left_boundary, bs_right_boundary
from src.network import create_layers, to_jax, n_parameters
from src.costs import trial_solution, kink_free_trial_solution, hard_cost, pinn_cost
from src.training import train, interior_points, initial_points, boundary_points, save_run

# Shared settings, the same for all networks so the comparison is fair. Will tune later
n_seeds = 5
sizes = [2, 30, 30, 1]
learning_rate = 1e-2
n_col, n_val = 1000, 500
w_initial = w_boundary = 100.0

# Points that are the same for every run: validation (own seed), and the initial line and boundaries for the PINN.
X_val = interior_points(n_val, -L, L, np.random.default_rng(SEED + 200))
X_initial = initial_points(200, -L, L)
X_left, X_right = boundary_points(100, -L), boundary_points(100, L)

# The two hard-coded networks: the same trial function form, with the naive g or the kink-free g.
u_naive = lambda layers, X: trial_solution(layers, X, bs_initial, bs_right_boundary, -L, L, tau_max)
u_kink_free = lambda layers, X: kink_free_trial_solution(layers, X, bs_initial, bs_left_boundary,
                                                         bs_right_boundary, -L, L, tau_max)


def pinn(layers, X_interior):
    return pinn_cost(layers, X_interior, X_initial, X_left, X_right, bs_initial, bs_left_boundary,
                     bs_right_boundary, tau_max, w_initial, w_boundary)


# Each method: its cost as a function of (weights, interior points), and the number of Adam steps.
methods = {
    "naive": (lambda layers, X: hard_cost(u_naive, layers, X, tau_max), 6000),
    "kink_free": (lambda layers, X: hard_cost(u_kink_free, layers, X, tau_max), 6000),
    "pinn": (pinn, 20000),
}

print(f"{n_seeds} seeds, network {sizes}, tanh, learning rate {learning_rate}, "
      f"{n_col} collocation and {n_val} validation points")
for name, (cost, n_iter) in methods.items():
    train_times, compile_times = [], []
    for i in range(n_seeds):
        layers = to_jax(create_layers(sizes, np.random.default_rng(SEED + i)))
        X_col = interior_points(n_col, -L, L, np.random.default_rng(SEED + 100 + i))
        params, history, timing = train(lambda p: cost(p, X_col), layers, n_iter=n_iter,
                                        learning_rate=learning_rate,
                                        validation_fn=jax.jit(lambda p: cost(p, X_val)))
        save_run(f"train_{name}_seed{i}", params, history, timing)
        train_times.append(timing["train"])
        compile_times.append(timing["compile"])
    print(f"{name:10s} {n_iter:6d} steps, training {np.mean(train_times):5.1f} ± {np.std(train_times):.1f} s, "
          f"compilation {np.mean(compile_times):.1f} s, {n_parameters(layers)} parameters")
