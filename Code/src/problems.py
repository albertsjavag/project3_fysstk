import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # must come before any JAX array is made, or jnp gives float32
import jax.numpy as jnp
from scipy.stats import norm

SEED = 2026  # base seed

K = 100.0      # strike
r = 0.05       # risk-free rate
sigma = 0.2    # volatility
T = 1.0        # expiry in years
L = 1.0        # cut-off: x in [-L, L]
t1 = 1 / 52    # time to expiry in years: one week, kink still sharp
t2 = 1.0       # time to expiry in years: one year, kink smoothed out

k = 2 * r / sigma**2         # the only combination of r and sigma left in the heat form
tau_max = sigma**2 * T / 2   # tau at t = 0, one year before expiry

# Explicit scheme: one fixed alpha = dtau / dx^2 for every grid. With this alpha, t1 is exactly one step on the
# coarsest grid (dx = 1/10); halving dx gives four times as many steps, so t1 and t2 = 52 t1 are hit on every grid.
dx_coarse = 0.1
alpha = sigma**2 / 2 * t1 / dx_coarse**2


def black_scholes_call(S, time_to_expiry, K=K, r=r, sigma=sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)
    payoff = np.maximum(S - K, 0.0)

    # The formula divides by zero at expiry: use a dummy time of 1 there, and pick the payoff below.
    t = np.where(time_to_expiry > 0, time_to_expiry, 1.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    price = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    return np.where(time_to_expiry > 0, price, payoff)


def to_heat_variables(S, time_to_expiry):
    x = np.log(S / K)
    tau = sigma**2 / 2 * time_to_expiry
    return x, tau


def from_heat_variables(x, tau):
    S = K * np.exp(x)
    time_to_expiry = tau * 2 / sigma**2
    return S, time_to_expiry


def price_to_u(C, x, tau):
    return C / K * np.exp((k - 1) * x / 2 + (k + 1)**2 * tau / 4)


def u_to_price(u, x, tau):
    return K * u * np.exp(-(k - 1) * x / 2 - (k + 1)**2 * tau / 4)


def sin_initial(x):
    return np.sin(np.pi * x)


def sin_exact(x, tau):
    return np.exp(-np.pi**2 * tau) * np.sin(np.pi * x)


# The three Black-Scholes conditions use jnp, so the networks can use them inside JAX code later.
def bs_initial(x):
    return jnp.maximum(jnp.exp((k + 1) * x / 2) - jnp.exp((k - 1) * x / 2), 0.0)


def bs_left_boundary(tau):
    return jnp.zeros_like(jnp.asarray(tau, dtype=float))


def bs_right_boundary(tau, kind="b"):
    # Far above the strike: (a) C = S, or (b) C = S - K exp(-r (T - t)), both written as u at x = L.
    stock = jnp.exp((k + 1) * L / 2 + (k + 1)**2 * tau / 4)
    if kind == "a":
        return stock
    return stock - jnp.exp((k - 1) * L / 2 + (k - 1)**2 * tau / 4)

def bs_exact(x, tau):
    # The Black-Scholes formula in heat form: to (S, T - t), price there, back to u.
    S, time_to_expiry = from_heat_variables(x, tau)
    C = black_scholes_call(S, time_to_expiry)
    return price_to_u(C, x, tau)
