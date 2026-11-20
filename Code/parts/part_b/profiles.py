import numpy as np
import matplotlib.pyplot as plt

from src.problems import L, K, t1, t2, to_heat_variables, u_to_price, bs_initial
from src.problems import bs_left_boundary, bs_right_boundary, bs_exact, sin_exact, sin_initial
from src.finite_difference import explicit_scheme
from src.plotting import save_fig

_, tau1 = to_heat_variables(K, t1)
_, tau2 = to_heat_variables(K, t2)

zero = lambda tau: 0.0
taus_sin = [0.02, 0.2]

for dx in (0.1, 0.01):
    x, u_fd = explicit_scheme(sin_initial, zero, zero, 0, 1, dx, taus_sin)
    for i, tau in enumerate(taus_sin):
        error = np.abs(u_fd[i] - sin_exact(x, tau))
        print(f"sin, dx = {dx}, tau = {tau}: max error = {error.max():.2e}")

taus_bs = [tau1, tau2]
labels = ["1 week", "1 year"]
runs = {}

for dx in (0.1, 0.01):
    x, u_fd = explicit_scheme(bs_initial, bs_left_boundary, bs_right_boundary, -L, L, dx, taus_bs)
    runs[dx] = (x, u_fd)

for dx, (x, u_fd) in runs.items():
    for i, tau in enumerate(taus_bs):
        u_exact = bs_exact(x, tau)
        error = np.abs(u_fd[i] - u_exact)
        j = np.argmax(error)
        C_error = np.abs(u_to_price(u_fd[i], x, tau) - u_to_price(u_exact, x, tau))
        j_C = np.argmax(C_error)
        print(f"{labels[i]}, dx = {dx}: max |u_fd - u_exact| = {error[j]:.2e} at x = {x[j]:+.2f}, "
              f"max price error = {C_error[j_C]:.2f} at S = {K * np.exp(x[j_C]):.0f}")

fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)

for i, ax in enumerate(axes):
    for dx, (x, u_fd) in runs.items():
        error = np.abs(u_fd[i] - bs_exact(x, taus_bs[i]))
        ax.semilogy(x, error, label=f"$\\Delta x = {dx}$")
    ax.set_ylim(1e-10, 1e-1)   # below the strike u is ~0 for both, so the error drops to 1e-280 and hides the rest
    ax.set_ylabel("$|u_{fd} - u_{exact}|$")
    ax.text(0.03, 0.9, labels[i], transform=ax.transAxes)

axes[-1].set_xlabel("$x$")
axes[0].legend()
save_fig(fig, "fd_profiles")


