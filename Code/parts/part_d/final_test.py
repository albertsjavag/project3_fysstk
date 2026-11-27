import json
import time

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.problems import (L, sigma, alpha, tau_max, bs_initial, bs_left_boundary, bs_right_boundary, bs_exact,
                          u_to_price)
from src.network import network, n_parameters
from src.costs import trial_solution, kink_free_trial_solution
from src.finite_difference import explicit_scheme
from src.training import load_run, RESULTS_DIR
from src.plotting import save_fig

n_seeds = 10
weeks = [1, 2, 4, 8, 13, 26, 52]
taus = [sigma**2 / 2 * w / 52 for w in weeks]
x_test = np.linspace(-L, L, 401)             # the test grid, used once, here
i_week, i_year = 0, len(weeks) - 1           # columns for 1 week and 1 year

with open(RESULTS_DIR / "final_configs.json") as f:
    configs = json.load(f)


def net_solution(method, activation):
    # The solution u(layers, X) of each kind of network, as in train_config.
    if method == "naive":
        return lambda layers, X: trial_solution(layers, X, bs_initial, bs_right_boundary, -L, L, tau_max, activation)
    if method == "kink_free":
        return lambda layers, X: kink_free_trial_solution(layers, X, bs_initial, bs_left_boundary, bs_right_boundary,
                                                          -L, L, tau_max, activation)
    return lambda layers, X: network(layers, X, activation)


def evaluate(u, layers, tau, x=x_test):
    X = jnp.column_stack([x, np.full_like(x, tau / tau_max)])     # column 1 is s = tau / tau_max
    return np.asarray(u(layers, X))


def error_measures(u_method, u_exact, x, tau):
    # Max |error|, RMSE and relative L2 error in u, and the max error in the price (same unit as K).
    error = np.abs(u_method - u_exact)
    price = np.abs(u_to_price(u_method, x, tau) - u_to_price(u_exact, x, tau)).max()
    return error.max(), np.sqrt(np.mean(error**2)), np.linalg.norm(error) / np.linalg.norm(u_exact), price


def r2(u_method, u_exact):
    return 1 - np.sum((u_method - u_exact)**2) / np.sum((u_exact - u_exact.mean())**2)


# --- The networks: one row per seed ---

measures = {}      # name -> seeds x times x (max, RMSE, relative L2, price)
extra = {}         # name -> seeds x (R² over the whole test grid, validation cost, training time)
for name, config in configs.items():
    u = net_solution(config["method"], config["activation"])
    measures[name] = np.zeros((n_seeds, len(taus), 4))
    extra[name] = np.zeros((n_seeds, 3))
    for i in range(n_seeds):
        layers, history, timing = load_run(f"final_{name}_seed{i}")
        u_net = np.array([evaluate(u, layers, tau) for tau in taus])
        u_exact = np.array([bs_exact(x_test, tau) for tau in taus])
        for j, tau in enumerate(taus):
            measures[name][i, j] = error_measures(u_net[j], u_exact[j], x_test, tau)
        validation = np.nanmin(history[:, 2])
        extra[name][i] = r2(u_net, u_exact), validation, timing["train"]
    configs[name]["n_parameters"] = n_parameters(layers)


# --- The scheme, as in part c (compare.py): plain on two grids, and with the kink removed ---

def ramp_heat_np(x, tau):
    w = np.sqrt(2 * tau) + 1e-12
    return x * norm.cdf(x / w) + w * norm.pdf(x / w)


v_initial = lambda x: np.asarray(bs_initial(x)) - np.maximum(x, 0.0)
v_left = lambda tau: 0.0 - ramp_heat_np(-L, tau)
v_right = lambda tau: float(bs_right_boundary(tau)) - ramp_heat_np(L, tau)


def run_scheme(dx, kink_free):
    if kink_free:
        x, v = explicit_scheme(v_initial, v_left, v_right, -L, L, dx, alpha, taus)
        return x, v + np.array([ramp_heat_np(x, tau) for tau in taus])     # put R back
    return explicit_scheme(bs_initial, bs_left_boundary, bs_right_boundary, -L, L, dx, alpha, taus)


fd_variants = {"scheme 1/10": (0.1, False), "scheme 1/100": (0.01, False), "kink-free scheme 1/10": (0.1, True)}
fd_times = {}
for name, (dx, kink_free) in fd_variants.items():
    run_scheme(dx, kink_free)                 # warm-up run, so first-call overhead is not timed
    t0 = time.perf_counter()
    x, u_fd = run_scheme(dx, kink_free)
    fd_times[name] = time.perf_counter() - t0
    u_exact = np.array([bs_exact(x, tau) for tau in taus])          # on the scheme's own grid x
    measures[name] = np.array([[error_measures(u_fd[j], u_exact[j], x, tau) for j, tau in enumerate(taus)]])
    extra[name] = np.array([[r2(u_fd, u_exact), np.nan, fd_times[name]]])


# --- Printed results: one table row per method ---

labels = {"kink_free": "kink-free trial", "kink_free_uniform": "kink-free, uniform points", "pinn": "PINN", "naive": "naive trial (control)",
          "relu": "kink-free ReLU (control)"}
for name, config in configs.items():
    print(f"{labels[name]}: {config['sizes']}, {config['activation']}, learning rate {config['learning_rate']:g}, "
          f"{config['n_col']} {config['placement']} points"
          + (f", w = {config['w_initial']:g}" if "w_initial" in config else "")
          + f", {config['n_parameters']} parameters")

order = list(fd_variants) + ["kink_free", "kink_free_uniform", "pinn", "naive", "relu"]
for label, j in [("1 week", i_week), ("1 year", i_year)]:
    print(f"--- {label} to expiry: mean ± std over {n_seeds} seeds (the scheme has no seeds) ---")
    for name in order:
        m = measures[name][:, j]
        print(f"{labels.get(name, name):26s} max {m[:, 0].mean():.1e} ± {m[:, 0].std():.0e}   "
              f"RMSE {m[:, 1].mean():.1e} ± {m[:, 1].std():.0e}   rel. L2 {m[:, 2].mean():.1e} ± {m[:, 2].std():.0e}   "
              f"price {m[:, 3].mean():.3f} ± {m[:, 3].std():.3f}")

print("--- Whole test grid (all times), validation cost and cost ---")
for name in order:
    e = extra[name]
    time_text = (f"run time {1e3 * e[:, 2].mean():.1f} ms" if name in fd_variants
                 else f"validation {e[:, 1].mean():.2e} ± {e[:, 1].std():.0e}   training {e[:, 2].mean():.0f} ± {e[:, 2].std():.0f} s")
    print(f"{labels.get(name, name):26s} R² {e[:, 0].mean():.6f}   {time_text}")

# --- Figure: max error against time to expiry, before (part c nets) and after tuning ---

# The part c networks (5 seeds, tanh, saved by parts/part_c/train.py), evaluated the same way.
part_c = {"kink_free": (net_solution("kink_free", "tanh"), "train_kink_free_seed{}"),
          "pinn": (net_solution("pinn", "tanh"), "train_pinn_seed{}")}
max_part_c = {}
for name, (u, run) in part_c.items():
    max_part_c[name] = np.zeros((5, len(taus)))
    for i in range(5):
        layers, _, _ = load_run(run.format(i))
        max_part_c[name][i] = [np.abs(evaluate(u, layers, tau) - bs_exact(x_test, tau)).max() for tau in taus]

fig, ax = plt.subplots(figsize=(3.4, 2.8))
for name, ls in [("scheme 1/10", "--"), ("kink-free scheme 1/10", ":")]:      # the scheme as reference, no seeds
    ax.plot(weeks, measures[name][0, :, 0], ls, color="0.35", label=name)
for name, color in [("kink_free", "tab:green"), ("pinn", "tab:blue")]:
    ax.plot(weeks, max_part_c[name].mean(axis=0), "--", color=color, alpha=0.6, label=f"{labels[name]}, part B")
    mean, std = measures[name][:, :, 0].mean(axis=0), measures[name][:, :, 0].std(axis=0)
    ax.plot(weeks, mean, "o-", ms=3, color=color, label=f"{labels[name]}, tuned")
    ax.fill_between(weeks, np.maximum(mean - std, 1e-12), mean + std, color=color, alpha=0.2, lw=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(weeks)
ax.set_xticklabels(["1w", "2w", "1m", "2m", "3m", "6m", "1y"])
ax.set_xlabel("time to expiry")
ax.set_ylabel(r"max $|u - u_{exact}|$")
fig.legend(fontsize=6, ncol=2, loc="outside lower center")
save_fig(fig, "tuning_vs_expiry")
