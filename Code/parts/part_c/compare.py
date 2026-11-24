import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.problems import (K, L, sigma, alpha, tau_max, bs_initial, bs_left_boundary, bs_right_boundary,
                          bs_exact, u_to_price)
from src.network import network
from src.costs import trial_solution, kink_free_trial_solution
from src.finite_difference import explicit_scheme
from src.training import load_run
from src.plotting import save_fig, PAGE

n_seeds = 5
weeks = [1, 2, 4, 8, 13, 26, 52]
taus = [sigma**2 / 2 * w / 52 for w in weeks]
x_test = np.linspace(-L, L, 401)
i_week, i_year = 0, len(weeks) - 1          # columns for 1 week and 1 year

nets = {
    "naive trial": lambda layers, X: trial_solution(layers, X, bs_initial, bs_right_boundary, -L, L, tau_max),
    "kink-free trial": lambda layers, X: kink_free_trial_solution(layers, X, bs_initial, bs_left_boundary,
                                                                  bs_right_boundary, -L, L, tau_max),
    "PINN": lambda layers, X: network(layers, X),
}
run_names = {"naive trial": "naive", "kink-free trial": "kink_free", "PINN": "pinn"}   # file names in results/
# One colour per method in every figure: grey/black for the scheme, colours for the networks.
colors = {"scheme 1/10": "black", "scheme 1/100": "0.55", "kink-free scheme 1/10": "tab:olive",
          "naive trial": "tab:red", "kink-free trial": "tab:green", "PINN": "tab:blue"}


def evaluate(u, layers, tau, x=x_test):
    X = jnp.column_stack([x, np.full_like(x, tau / tau_max)])     # column 1 is s = tau / tau_max
    return np.asarray(u(layers, X))


# --- Errors of the networks: one row per seed, one column per time ---

max_errors, rmse, price_errors, train_times = {}, {}, {}, {}

for name, u in nets.items():
    max_errors[name] = np.zeros((n_seeds, len(taus)))
    rmse[name] = np.zeros((n_seeds, len(taus)))
    price_errors[name] = np.zeros((n_seeds, len(taus)))
    train_times[name] = []
    for i in range(n_seeds):
        layers, history, timing = load_run(f"train_{run_names[name]}_seed{i}")
        train_times[name].append(timing["train"])
        for j, tau in enumerate(taus):
            u_net = evaluate(u, layers, tau)
            u_exact = bs_exact(x_test, tau)
            error = np.abs(u_net - u_exact)
            max_errors[name][i, j] = error.max()
            rmse[name][i, j] = np.sqrt(np.mean(error**2))
            price_errors[name][i, j] = np.abs(u_to_price(u_net, x_test, tau) - u_to_price(u_exact, x_test, tau)).max()


# --- The scheme: plain on two grids, and with the kink removed (v = u - R) for a fair comparison ---

def ramp_heat_np(x, tau):
    # Same formula as ramp_heat in problems.py, in plain numpy: called every time step, and JAX call overhead
    # would otherwise dominate the run time of the scheme.
    w = np.sqrt(2 * tau) + 1e-12
    return x * norm.cdf(x / w) + w * norm.pdf(x / w)


v_initial = lambda x: np.asarray(bs_initial(x)) - np.maximum(x, 0.0)
v_left = lambda tau: 0.0 - ramp_heat_np(-L, tau)
v_right = lambda tau: float(bs_right_boundary(tau)) - ramp_heat_np(L, tau)


def run_scheme(dx, kink_free, taus):
    if kink_free:
        x, v = explicit_scheme(v_initial, v_left, v_right, -L, L, dx, alpha, taus)
        return x, v + np.array([ramp_heat_np(x, tau) for tau in taus])     # put R back
    return explicit_scheme(bs_initial, bs_left_boundary, bs_right_boundary, -L, L, dx, alpha, taus)


fd_variants = {"scheme 1/10": (0.1, False), "scheme 1/100": (0.01, False), "kink-free scheme 1/10": (0.1, True)}
fd_solutions, fd_times = {}, {}

for name, (dx, kink_free) in fd_variants.items():
    run_scheme(dx, kink_free, taus)          # warm-up run, so first-call overhead is not timed
    t0 = time.perf_counter()
    x, u_fd = run_scheme(dx, kink_free, taus)
    fd_times[name] = time.perf_counter() - t0
    fd_solutions[name] = (x, u_fd)

    # One row only: the scheme has no seeds, so its std below is 0.
    max_errors[name] = np.zeros((1, len(taus)))
    rmse[name] = np.zeros((1, len(taus)))
    price_errors[name] = np.zeros((1, len(taus)))
    for j, tau in enumerate(taus):
        u_exact = bs_exact(x, tau)                              # on the scheme's own grid x
        error = np.abs(u_fd[j] - u_exact)
        max_errors[name][0, j] = error.max()
        rmse[name][0, j] = np.sqrt(np.mean(error**2))
        price_errors[name][0, j] = np.abs(u_to_price(u_fd[j], x, tau) - u_to_price(u_exact, x, tau)).max()


# --- Printed results ---

order = ["scheme 1/10", "scheme 1/100", "kink-free scheme 1/10", "naive trial", "kink-free trial", "PINN"]
for label, j in [("1 week", i_week), ("1 year", i_year)]:
    print(f"--- {label} to expiry: max |error| and RMSE in u (mean ± std over {n_seeds} seeds), max price error ---")
    for name in order:
        m, r, c = max_errors[name][:, j], rmse[name][:, j], price_errors[name][:, j]
        print(f"{name:22s} max {m.mean():.1e} ± {m.std():.0e}   RMSE {r.mean():.1e} ± {r.std():.0e}   "
              f"price {c.mean():.3f} kr")

print("--- Cost ---")
for name in fd_variants:
    print(f"{name:22s} run time {1e3 * fd_times[name]:.1f} ms for all {len(taus)} times")
layers, _, _ = load_run("train_kink_free_seed0")
t0 = time.perf_counter()
for tau in taus:
    evaluate(nets["kink-free trial"], layers, tau)
evaluation_time = time.perf_counter() - t0
for name in nets:
    print(f"{name:22s} training {np.mean(train_times[name]):.1f} ± {np.std(train_times[name]):.1f} s")
print(f"evaluating a trained net at {len(x_test)} x {len(taus)} points: {1e3 * evaluation_time:.0f} ms")

# A stock price that is not a grid point of the scheme: the nets are evaluated there, the scheme interpolates.
S_off = 103.7
x_off = np.array([np.log(S_off / K)])
exact_off = bs_exact(x_off, taus[i_week])[0]
print(f"--- Error at S = {S_off} (not on the scheme grid), 1 week to expiry ---")
for name, (x, u_fd) in fd_solutions.items():
    if fd_variants[name][1]:
        # Kink-free: interpolate only the smooth part v = u - R, and add the ramp R exactly at the new point.
        R_grid, R_off = ramp_heat_np(x, taus[i_week]), ramp_heat_np(x_off[0], taus[i_week])
        u_off = np.interp(x_off[0], x, u_fd[i_week] - R_grid) + R_off
    else:
        u_off = np.interp(x_off[0], x, u_fd[i_week])
    print(f"{name:22s} {abs(u_off - exact_off):.1e}   (linear interpolation)")
for name, u in nets.items():
    layers, _, _ = load_run(f"train_{run_names[name]}_seed0")
    print(f"{name:22s} {abs(evaluate(u, layers, taus[i_week], x=x_off)[0] - exact_off):.1e}   (seed 0)")


# --- Figure 1: error against x at 1 week and 1 year ---

fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True)
for ax, j, label in [(axes[0], i_week, "1 week"), (axes[1], i_year, "1 year")]:
    # Only the coarse scheme here, to keep the panels readable; the other variants are in error_vs_expiry and the table.
    x, u_fd = fd_solutions["scheme 1/10"]
    ax.semilogy(x, np.abs(u_fd[j] - bs_exact(x, taus[j])), "o-", ms=2, color=colors["scheme 1/10"], label="scheme 1/10")
    for name, u in nets.items():
        layers, _, _ = load_run(f"train_{run_names[name]}_seed0")
        ax.semilogy(x_test, np.abs(evaluate(u, layers, taus[j]) - bs_exact(x_test, taus[j])), color=colors[name],
                    label=name)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_ylabel(r"$|u - u_{exact}|$")
    ax.text(0.03, 0.88, label, transform=ax.transAxes)
axes[1].set_xlabel(r"$x$ (0 = strike)")
handles, labels = axes[0].get_legend_handles_labels()            # one entry per method, not one per panel
fig.legend(handles, labels, fontsize=6, ncol=2, loc="outside lower center")
save_fig(fig, "methods_error_x")


# --- Figure 2: max error against time to expiry ---

fig, ax = plt.subplots(figsize=(3.4, 3.2))
for name in order:
    m = max_errors[name]
    style = "--" if name in fd_variants else "-"                 # dashed: scheme, solid: networks
    ax.errorbar(weeks, m.mean(axis=0), yerr=m.std(axis=0), fmt="o" + style, ms=3, capsize=2,
                color=colors[name], label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(weeks)
ax.set_xticklabels(["1w", "2w", "1m", "2m", "3m", "6m", "1y"])
ax.set_xlabel("time to expiry")
ax.set_ylabel(r"max $|u - u_{exact}|$")
fig.legend(fontsize=6, ncol=2, loc="outside lower center")
save_fig(fig, "error_vs_expiry")


# --- Figure 3: error over the whole domain, every week from 1 to 52, one panel per method ---

weeks_map = np.arange(1, 53)
taus_map = sigma**2 / 2 * weeks_map / 52
fig, axes = plt.subplots(2, 3, figsize=(PAGE[0], 4.2), sharex=True, sharey=True)
panels = []
for name, (dx, kink_free) in fd_variants.items():
    x, u_fd = run_scheme(dx, kink_free, list(taus_map))
    panels.append((name, x, np.abs(u_fd - np.array([bs_exact(x, tau) for tau in taus_map]))))
for name, u in nets.items():
    layers, _, _ = load_run(f"train_{run_names[name]}_seed0")
    error = np.array([np.abs(evaluate(u, layers, tau) - bs_exact(x_test, tau)) for tau in taus_map])
    panels.append((name, x_test, error))
for ax, (name, x, error) in zip(axes.flat, panels):
    mesh = ax.pcolormesh(x, weeks_map, np.log10(np.maximum(error, 1e-12)), vmin=-6, vmax=-1, shading="auto")
    ax.set_yscale("log")
    ax.text(0.03, 0.9, name, transform=ax.transAxes, fontsize=8, color="white")
for ax in axes[1]:
    ax.set_xlabel(r"$x$")
for ax in axes[:, 0]:
    ax.set_ylabel("weeks to expiry")
fig.colorbar(mesh, ax=axes, label=r"$\log_{10}|u - u_{exact}|$")
save_fig(fig, "error_maps")
