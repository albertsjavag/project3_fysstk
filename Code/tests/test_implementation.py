import numpy as np
import pytest

from src.problems import (K, r, L, alpha, tau_max, black_scholes_call, to_heat_variables, from_heat_variables,
                          price_to_u, u_to_price, sin_initial, sin_exact, bs_initial, bs_left_boundary,
                          bs_right_boundary, bs_exact, ramp_heat)
from src.finite_difference import explicit_scheme
from src.network import create_layers, to_jax
from src.costs import kink_free_trial_solution
import jax.numpy as jnp

zero = lambda tau: 0.0


# --- The Black-Scholes formula ---

def test_price_at_the_money_one_year():
    # S = K = 100, r = 0.05, sigma = 0.2, one year: the textbook price is 10.4506.
    assert black_scholes_call(100.0, 1.0) == pytest.approx(10.4506, abs=1e-4)


def test_price_at_expiry_is_payoff():
    S = np.array([80.0, 100.0, 120.0])
    assert np.allclose(black_scholes_call(S, 0.0), [0.0, 0.0, 20.0])


def test_price_is_below_stock_price():
    # A call can never be worth more than the stock itself.
    S = np.array([80.0, 100.0, 120.0])
    assert np.all(black_scholes_call(S, 1.0) < S)


# --- The change of variables ---

def test_stock_price_and_time_survive_round_trip():
    x, tau = to_heat_variables(120.0, 0.5)
    S, time_to_expiry = from_heat_variables(x, tau)
    assert S == pytest.approx(120.0)
    assert time_to_expiry == pytest.approx(0.5)


def test_option_price_survives_round_trip():
    u = price_to_u(15.0, 0.3, 0.01)
    assert u_to_price(u, 0.3, 0.01) == pytest.approx(15.0)


def test_strike_is_at_x_zero():
    x, tau = to_heat_variables(K, 1.0)
    assert x == pytest.approx(0.0)
    assert tau == pytest.approx(tau_max)


# --- Conditions and exact solutions ---

def test_bs_exact_starts_at_payoff():
    x = np.linspace(-L, L, 21)
    assert np.allclose(bs_exact(x, 0.0), bs_initial(x))


def test_sin_exact_starts_at_sin():
    x = np.linspace(0, 1, 11)
    assert np.allclose(sin_exact(x, 0.0), sin_initial(x))


def test_right_boundary_b_agrees_with_payoff_in_the_corner():
    assert float(bs_right_boundary(0.0, kind="b")) == pytest.approx(float(bs_initial(L)))


def test_left_boundary_is_zero():
    assert float(bs_left_boundary(0.01)) == 0.0


def time_and_space_derivatives(u, x, tau):
    # u_tau and u_xx with central differences. Fine for testing the exact solutions; the networks use autodiff.
    h = 1e-4
    u_tau = (u(x, tau + h) - u(x, tau - h)) / (2 * h)
    u_xx = (u(x + h, tau) - 2 * u(x, tau) + u(x - h, tau)) / h**2
    return u_tau, u_xx


def test_bs_exact_solves_heat_equation():
    u_tau, u_xx = time_and_space_derivatives(bs_exact, np.array([-0.5, 0.0, 0.5]), 0.01)
    assert np.allclose(u_tau, u_xx, rtol=1e-3)


def test_sin_exact_solves_heat_equation():
    u_tau, u_xx = time_and_space_derivatives(sin_exact, np.array([0.25, 0.5, 0.75]), 0.01)
    assert np.allclose(u_tau, u_xx, rtol=1e-3)


# --- The explicit scheme ---

def test_scheme_error_falls_by_four_when_dx_is_halved():
    # Second order: half the grid spacing gives a quarter of the error.
    x, u_coarse = explicit_scheme(sin_initial, zero, zero, 0, 1, 0.1, alpha, [0.02])
    error_coarse = np.abs(u_coarse[0] - sin_exact(x, 0.02)).max()
    x, u_fine = explicit_scheme(sin_initial, zero, zero, 0, 1, 0.05, alpha, [0.02])
    error_fine = np.abs(u_fine[0] - sin_exact(x, 0.02)).max()
    assert error_coarse / error_fine == pytest.approx(4, rel=0.1)


def test_scheme_gives_same_answer_whatever_times_we_ask_for():
    x, u_end_only = explicit_scheme(bs_initial, bs_left_boundary, bs_right_boundary, -L, L, 0.1, alpha, [tau_max])
    x, u_with_stops = explicit_scheme(bs_initial, bs_left_boundary, bs_right_boundary, -L, L, 0.1, alpha,
                                      [tau_max / 52, tau_max])
    assert np.array_equal(u_end_only[0], u_with_stops[1])


def test_scheme_refuses_unstable_alpha():
    with pytest.raises(ValueError):
        explicit_scheme(sin_initial, zero, zero, 0, 1, 0.1, 0.6, [0.02])


def test_scheme_refuses_time_between_two_steps():
    with pytest.raises(ValueError):
        explicit_scheme(sin_initial, zero, zero, 0, 1, 0.1, alpha, [1e-4])


# --- The kink-free trial function ---

def test_ramp_heat_starts_as_ramp():
    x = np.array([-0.5, 0.0, 0.5])
    assert np.allclose(ramp_heat(x, 0.0), [0.0, 0.0, 0.5])


def test_ramp_heat_solves_heat_equation():
    u_tau, u_xx = time_and_space_derivatives(lambda x, tau: np.asarray(ramp_heat(x, tau)), np.array([-0.1, 0.0, 0.1]), 0.01)
    assert np.allclose(u_tau, u_xx, rtol=1e-3)


def test_kink_free_trial_satisfies_conditions_for_any_weights():
    layers = to_jax(create_layers([2, 10, 1], np.random.default_rng(0)))   # untrained network
    u = lambda X: kink_free_trial_solution(layers, X, bs_initial, bs_left_boundary, bs_right_boundary, -L, L, tau_max)
    x, s = jnp.linspace(-L, L, 5), jnp.linspace(0, 1, 5)
    assert np.allclose(u(jnp.column_stack([x, jnp.zeros(5)])), bs_initial(x))                         # start
    assert np.allclose(u(jnp.column_stack([-L * jnp.ones(5), s])), 0.0)                                 # left
    assert np.allclose(u(jnp.column_stack([L * jnp.ones(5), s])), bs_right_boundary(s * tau_max))       # right
