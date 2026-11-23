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
