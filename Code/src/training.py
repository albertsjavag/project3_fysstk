import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def interior_points(n, x_min, x_max, rng):
    x = rng.uniform(x_min, x_max, n)
    s = rng.uniform(0, 1, n)
    return jnp.asarray(np.column_stack([x, s]))

def initial_points(n, x_min, x_max):
    x = np.linspace(x_min, x_max, n)
    s = np.zeros(n)
    return jnp.asarray(np.column_stack([x, s]))

def boundary_points(n, x_edge):
    x = np.full(n, x_edge)
    s = np.linspace(0, 1, n)
    return jnp.asarray(np.column_stack([x, s]))

# Training:

def train(cost_fn, params, n_iter=2000, learning_rate=1e-2, every=50, validation_fn=None):
    # Adam (from adam_minimise, week42monday cell 26). Returns the parameters with the lowest validation value
    # when validation_fn is given, else the last ones; the history (iteration, cost, validation); and the time.
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(params, m, v, it):
        cost, grad = jax.value_and_grad(cost_fn)(params)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, grad)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_**2, v, grad)
        params = jax.tree_util.tree_map(
            lambda p, m_, v_: p - learning_rate * (m_ / (1 - b1**it)) / (jnp.sqrt(v_ / (1 - b2**it)) + eps),
            params, m, v)
        return params, m, v, cost

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    # The first calls compile step (and the validation); time them on their own, so the training time below
    # is training only.
    t0 = time.perf_counter()
    jax.block_until_ready(step(params, m, v, 1.0))
    if validation_fn is not None:
        jax.block_until_ready(validation_fn(params))
    compile_time = time.perf_counter() - t0

    history = []
    best_params, best_validation = params, np.inf
    t0 = time.perf_counter()
    for it in range(1, n_iter + 1):
        params, m, v, cost = step(params, m, v, float(it))
        if it == 1 or it % every == 0 or it == n_iter:
            row = [it, float(cost)]
            if validation_fn is not None:
                row.append(float(validation_fn(params)))
                if row[-1] < best_validation:
                    best_params, best_validation = params, row[-1]
            history.append(row)
    jax.block_until_ready(params)
    timing = {"compile": compile_time, "train": time.perf_counter() - t0, "steps": n_iter}

    if validation_fn is None:
        best_params = params
    return best_params, np.array(history), timing


# --- Saving and loading a trained network ---

def save_run(name, layers, history, timing):
    # One .npz file in Code/results/ with the weights (W0, b0, W1, b1, ...), the history and the timing.
    RESULTS_DIR.mkdir(exist_ok=True)
    weights = {}
    for i, (W, b) in enumerate(layers):
        weights[f"W{i}"], weights[f"b{i}"] = np.asarray(W), np.asarray(b)
    np.savez(RESULTS_DIR / f"{name}.npz", history=history, **weights,
             compile=timing["compile"], train=timing["train"], steps=timing["steps"])


def load_run(name):
    stored = np.load(RESULTS_DIR / f"{name}.npz")
    n_layers = sum(key.startswith("W") for key in stored.files)
    layers = [(jnp.asarray(stored[f"W{i}"]), jnp.asarray(stored[f"b{i}"])) for i in range(n_layers)]
    timing = {"compile": float(stored["compile"]), "train": float(stored["train"]), "steps": int(stored["steps"])}
    return layers, stored["history"], timing


# --- One training run on the Black-Scholes problem, for the scans in part d ---

def kink_points(n, x_min, x_max, rng):
    # Half of the points uniform in the domain, half close to the kink: |x| < 0.2 and the first tenth of the time.
    n_near = n // 2
    near = np.column_stack([rng.uniform(-0.2, 0.2, n_near), rng.uniform(0, 0.1, n_near)])
    return jnp.vstack([interior_points(n - n_near, x_min, x_max, rng), jnp.asarray(near)])


def train_config(method, seed_offset=0, sizes=(2, 30, 30, 1), activation="tanh", learning_rate=1e-2, n_iter=None,
                 n_col=1000, placement="uniform", w_initial=100.0, w_boundary=100.0, name=None):
    # method: "naive" or "kink_free" (hard-coded trial networks) or "pinn". Weights are drawn with seed
    # SEED + seed_offset and the collocation points with SEED + 100 + seed_offset, so each source can be
    # varied on its own; the validation points (SEED + 200) are the same for every run.
    from src.problems import SEED, L, tau_max, bs_initial, bs_left_boundary, bs_right_boundary
    from src.network import create_layers, to_jax, network, n_parameters
    from src.costs import trial_solution, kink_free_trial_solution, hard_cost, pinn_cost

    if n_iter is None:
        n_iter = 20000 if method == "pinn" else 6000
    rng_points = np.random.default_rng(SEED + 100 + seed_offset)
    if placement == "uniform":
        X_col = interior_points(n_col, -L, L, rng_points)
    else:
        X_col = kink_points(n_col, -L, L, rng_points)
    X_val = interior_points(500, -L, L, np.random.default_rng(SEED + 200))
    layers = to_jax(create_layers(list(sizes), np.random.default_rng(SEED + seed_offset), activation))

    if method == "pinn":
        X_initial = initial_points(200, -L, L)
        X_left, X_right = boundary_points(100, -L), boundary_points(100, L)

        def cost(p, X_interior, w0, wb):
            return pinn_cost(p, X_interior, X_initial, X_left, X_right, bs_initial, bs_left_boundary,
                             bs_right_boundary, tau_max, w0, wb, activation)

        cost_fn = lambda p: cost(p, X_col, w_initial, w_boundary)
        # Validation without the weights, so runs with different weights can be compared on the same scale.
        validation_fn = jax.jit(lambda p: cost(p, X_val, 1.0, 1.0))
    else:
        if method == "naive":
            u = lambda p, X: trial_solution(p, X, bs_initial, bs_right_boundary, -L, L, tau_max, activation)
        else:
            u = lambda p, X: kink_free_trial_solution(p, X, bs_initial, bs_left_boundary, bs_right_boundary,
                                                      -L, L, tau_max, activation)
        cost_fn = lambda p: hard_cost(u, p, X_col, tau_max)
        validation_fn = jax.jit(lambda p: hard_cost(u, p, X_val, tau_max))

    params, history, timing = train(cost_fn, layers, n_iter=n_iter, learning_rate=learning_rate,
                                    validation_fn=validation_fn)
    if name is not None:
        save_run(name, params, history, timing)
    return {"params": params, "history": history, "timing": timing, "validation": float(history[:, 2].min()),
            "n_parameters": n_parameters(params)}
