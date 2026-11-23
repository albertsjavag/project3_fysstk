import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.network import network
from src.problems import ramp_heat


def d_dxk(fun, k):
    # From week43monday (cell 5): the derivative of fun(layers, X) with respect to input column k, row by row.
    # Nest it for higher derivatives: d_dxk(d_dxk(u, 0), 0) is the second derivative in column 0.
    def wrapped(layers, X):
        def one_row(xk, row):
            return fun(layers, row.at[k].set(xk)[None, :])[0]
        return jax.vmap(jax.grad(one_row), in_axes=(0, 0))(X[:, k], X)
    return wrapped


def trial_solution(layers, X, u_initial, right_boundary, x_min, x_max, tau_end, activation="tanh"):
    # Column 0 of X is x, column 1 is the scaled time s = tau / tau_end in [0, 1].
    x, s = X[:, 0], X[:, 1]
    tau = s * tau_end

    # g satisfies the conditions: the initial condition, plus a straight line in x that lifts the right end
    # as the right boundary moves. Assumes the left boundary is constant and equal to u_initial(x_min).
    g = u_initial(x) + (x - x_min) / (x_max - x_min) * (right_boundary(tau) - right_boundary(0.0))

    # Zero at both ends and at s = 0, so the network cannot change the conditions.
    bubble = (x - x_min) * (x_max - x) * s
    return g + bubble * network(layers, X, activation)


def kink_free_g(x, tau, u_initial, left_boundary, right_boundary, x_min, x_max):
    # Like g above, but the kink of the payoff at x = 0 (slope 0 -> 1) is replaced by the heat-smoothed ramp,
    # which solves the heat equation exactly. What is left for the network is smooth.
    smooth_part = lambda x: u_initial(x) - jnp.maximum(x, 0.0)          # payoff without its kink
    raw = lambda x: ramp_heat(x, tau) + smooth_part(x)                    # equals u_initial at tau = 0
    # Straight lines in x that move both ends onto the boundary values (both are zero at tau = 0).
    right_fix = (x - x_min) / (x_max - x_min) * (right_boundary(tau) - raw(x_max))
    left_fix = (x_max - x) / (x_max - x_min) * (left_boundary(tau) - raw(x_min))
    return raw(x) + right_fix + left_fix


def kink_free_trial_solution(layers, X, u_initial, left_boundary, right_boundary, x_min, x_max, tau_end,
                             activation="tanh"):
    x, s = X[:, 0], X[:, 1]
    g = kink_free_g(x, s * tau_end, u_initial, left_boundary, right_boundary, x_min, x_max)
    bubble = (x - x_min) * (x_max - x) * s
    return g + bubble * network(layers, X, activation)


def residual(u, layers, X, tau_end):
    # u_tau - u_xx. The network sees s = tau / tau_end, so u_tau = u_s / tau_end (chain rule).
    u_s = d_dxk(u, 1)(layers, X)
    u_xx = d_dxk(d_dxk(u, 0), 0)(layers, X)
    return u_s / tau_end - u_xx


def hard_cost(u, layers, X_col, tau_end):
    # The trial function satisfies the conditions, so only the equation is in the cost.
    return jnp.mean(residual(u, layers, X_col, tau_end) ** 2)


def pinn_cost(layers, X_col, X_initial, X_left, X_right, u_initial, left_boundary, right_boundary,
              tau_end, w_initial, w_boundary, activation="tanh"):
    # The network itself is the solution, so the equation and the conditions are all in the cost.
    u = lambda layers, X: network(layers, X, activation)

    residual_term = jnp.mean(residual(u, layers, X_col, tau_end) ** 2)
    initial_term = jnp.mean((u(layers, X_initial) - u_initial(X_initial[:, 0])) ** 2)
    left_term = jnp.mean((u(layers, X_left) - left_boundary(X_left[:, 1] * tau_end)) ** 2)
    right_term = jnp.mean((u(layers, X_right) - right_boundary(X_right[:, 1] * tau_end)) ** 2)

    return residual_term + w_initial * initial_term + w_boundary * (left_term + right_term)
