import numpy as np
from scipy.stats import norm

SEED = 2026  # base seed

K = 100.0      # strike
r = 0.05       # risk-free rate
sigma = 0.2    # volatility
T = 1.0        # expiry in years
L = 1.0        # cut-off: x in [-L, L]


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

