import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.problems import (L, SEED, sigma, tau_max, t1, t2, bs_initial, bs_left_boundary, bs_right_boundary,
                          bs_exact, ramp_heat)
from src.network import create_layers, to_jax
from src.costs import trial_solution, kink_free_g
from src.plotting import save_fig

# The two g's, i.e. the trial functions with the network switched off (all weights zero).
zero_net = jax.tree_util.tree_map(jnp.zeros_like, to_jax(create_layers([2, 30, 30, 1], np.random.default_rng(SEED))))


def g_naive(x, tau):
    X = jnp.column_stack([x, np.full_like(x, tau / tau_max)])
    return np.asarray(trial_solution(zero_net, X, bs_initial, bs_right_boundary, -L, L, tau_max))


def g_kink_free(x, tau):
    return np.asarray(kink_free_g(x, tau, bs_initial, bs_left_boundary, bs_right_boundary, -L, L))


# How far each g alone (N = 0) is from the exact solution: what the network has to add.
x = np.linspace(-L, L, 401)
print("max |u_exact - g| with the network switched off (N = 0)")
for label, t in [("1 week", t1), ("1 year", t2)]:
    tau = sigma**2 / 2 * t
    print(f"{label:7s} naive g {np.abs(bs_exact(x, tau) - g_naive(x, tau)).max():.1e}   "
          f"kink-free g {np.abs(bs_exact(x, tau) - g_kink_free(x, tau)).max():.1e}")

# Figure: near the strike, one week before expiry.
tau = sigma**2 / 2 * t1
x = np.linspace(-0.15, 0.15, 601)
u_exact = bs_exact(x, tau)

fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True)
axes[0].plot(x, np.asarray(bs_initial(x)), "k--", label="payoff (kink)")
axes[0].plot(x, u_exact, color="tab:blue", label="exact, 1 week")
axes[0].plot(x, np.asarray(ramp_heat(x, tau)), color="tab:green", ls=":", label="heat-smoothed ramp $R$")
axes[0].set_ylabel(r"$u$")
axes[0].legend(fontsize=7, loc="upper left")

# What the network must learn: the exact solution minus g (the label says the quantity, the caption its meaning).
axes[1].plot(x, u_exact - g_naive(x, tau), color="tab:red", label="naive $g$")
axes[1].plot(x, u_exact - g_kink_free(x, tau), color="tab:green", label="kink-free $g$")
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].set_ylabel(r"$u_{exact} - g$")
axes[1].set_xlabel(r"$x$ (0 = strike)")
axes[1].legend(fontsize=7, loc="upper left")
save_fig(fig, "kink_decomposition")
