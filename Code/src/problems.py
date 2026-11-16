"""
The two problems, both written as u_tau = u_xx on an interval, and the common seed.

- The test case of the project text: u(x, 0) = sin(pi x), u = 0 at both ends,
  exact solution exp(-pi^2 t) sin(pi x).
- A European call option: the Black-Scholes equation after the change of
  variables, with the kinked payoff as initial condition on a cut-off domain
  [-L, L]. Exact solution from the Black-Scholes formula.

Also the Black-Scholes formula in the original variables (S, t), the change of
variables both ways, and synthetic option prices with noise for the volatility
extension.
"""

SEED = 2026  # base seed; every run derives its own seeds from this one
