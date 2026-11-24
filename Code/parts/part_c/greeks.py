import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.problems import (K, L, r, sigma, k, tau_max, t1, alpha, bs_initial, bs_left_boundary,
                          bs_right_boundary)
from src.network import network
from src.costs import trial_solution, kink_free_trial_solution
from src.plotting import save_fig
from src.training import load_run
from src.finite_difference import explicit_scheme

n_seeds = 5
time_to_expiry = t1                       # one week, where Gamma is sharpest
tau = sigma**2 / 2 * time_to_expiry
S = np.linspace(80, 120, 201)             # zoom in around the strike

nets = {
    "naive trial": lambda layers, X: trial_solution(layers, X, bs_initial, bs_right_boundary, -L, L, tau_max),
    "kink-free trial": lambda layers, X: kink_free_trial_solution(layers, X, bs_initial, bs_left_boundary,
                                                                  bs_right_boundary, -L, L, tau_max),
    "PINN": lambda layers, X: network(layers, X),
}
run_names = {"naive trial": "naive", "kink-free trial": "kink_free", "PINN": "pinn"}
colors = {"scheme 1/10": "black", "scheme 1/100": "0.55", "naive trial": "tab:red", "kink-free trial": "tab:green",
          "PINN": "tab:blue"}


def exact_greeks(S, time_to_expiry):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(time_to_expiry))
    return delta, gamma


def net_greeks(u, layers, S, tau):
    def price(S_one):                                       # the price C for one stock price
        x = jnp.log(S_one / K)
        X = jnp.array([[x, tau / tau_max]])
        u_value = u(layers, X)[0]
        return K * u_value * jnp.exp(-(k - 1) * x / 2 - (k + 1)**2 * tau / 4)   # u back to price
    delta = jax.vmap(jax.grad(price))(jnp.asarray(S))
    gamma = jax.vmap(jax.grad(jax.grad(price)))(jnp.asarray(S))
    return np.asarray(delta), np.asarray(gamma)


def scheme_greeks(dx):
    # Finite differences on the scheme's grid, which is uniform in x = ln(S/K): differentiate in x, then
    # convert to S with the chain rule, dC/dS = C_x / S and d2C/dS2 = (C_xx - C_x) / S^2.
    x, u_fd = explicit_scheme(bs_initial, bs_left_boundary, bs_right_boundary, -L, L, dx, alpha, [tau])
    C = K * u_fd[0] * np.exp(-(k - 1) * x / 2 - (k + 1)**2 * tau / 4)
    C_x = np.gradient(C, x)
    C_xx = np.gradient(C_x, x)
    S_grid = K * np.exp(x)
    inside = (S_grid >= S[0]) & (S_grid <= S[-1])           # only the grid points in the plotted range
    return S_grid[inside], (C_x / S_grid)[inside], ((C_xx - C_x) / S_grid**2)[inside]


delta_exact, gamma_exact = exact_greeks(S, time_to_expiry)

# --- Networks: max error in Delta and Gamma for S in [80, 120], every seed ---

delta_errors, gamma_errors = {}, {}
for name, u in nets.items():
    delta_errors[name], gamma_errors[name] = [], []
    for i in range(n_seeds):
        layers, _, _ = load_run(f"train_{run_names[name]}_seed{i}")
        delta, gamma = net_greeks(u, layers, S, tau)
        delta_errors[name].append(np.abs(delta - delta_exact).max())
        gamma_errors[name].append(np.abs(gamma - gamma_exact).max())

# --- Scheme: the same errors, at its own grid points ---

schemes = {"scheme 1/10": 0.1, "scheme 1/100": 0.01}
scheme_results = {}
for name, dx in schemes.items():
    S_grid, delta, gamma = scheme_greeks(dx)
    delta_ref, gamma_ref = exact_greeks(S_grid, time_to_expiry)
    scheme_results[name] = (S_grid, delta, gamma)
    delta_errors[name] = [np.abs(delta - delta_ref).max()]
    gamma_errors[name] = [np.abs(gamma - gamma_ref).max()]

# --- Printed results ---

i_strike = np.argmin(np.abs(S - K))
print(f"Greeks one week before expiry, S in [{S[0]:.0f}, {S[-1]:.0f}]; exact at S = K: "
      f"Delta {delta_exact[i_strike]:.4f}, Gamma {gamma_exact[i_strike]:.4f}")
print("max |error| (mean ± std over seeds for the networks)")
for name in ["scheme 1/10", "scheme 1/100", "naive trial", "kink-free trial", "PINN"]:
    d, g = np.array(delta_errors[name]), np.array(gamma_errors[name])
    print(f"{name:16s} Delta {d.mean():.1e} ± {d.std():.0e}   Gamma {g.mean():.1e} ± {g.std():.0e}")

# --- Figure: Delta and Gamma against S, seed 0 for the networks ---

fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True)
axes[0].plot(S, delta_exact, color="0.75", lw=3, label="exact")
axes[1].plot(S, gamma_exact, color="0.75", lw=3)
S_grid, delta, gamma = scheme_results["scheme 1/10"]
axes[0].plot(S_grid, delta, "o", color=colors["scheme 1/10"], ms=4, label="scheme 1/10")
axes[1].plot(S_grid, gamma, "o", color=colors["scheme 1/10"], ms=4)
for name, u in nets.items():
    layers, _, _ = load_run(f"train_{run_names[name]}_seed0")
    delta, gamma = net_greeks(u, layers, S, tau)
    axes[0].plot(S, delta, color=colors[name], label=name)
    axes[1].plot(S, gamma, color=colors[name])
axes[0].set_ylabel(r"Delta $\partial C/\partial S$")
axes[1].set_ylabel(r"Gamma $\partial^2 C/\partial S^2$")
axes[1].set_xlabel(r"stock price $S$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6, ncol=3, loc="outside lower center")
save_fig(fig, "greeks")
