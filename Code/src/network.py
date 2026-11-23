"""
The feed-forward network N(x, t; theta) in JAX: initialization of the weights
from a seed, the forward pass and the activation functions. The derivatives
u_t and u_xx of a network-based solution by automatic differentiation, and the
residual u_t - u_xx at a set of points.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def create_layers(sizes, rng, hidden_activation="sigmoid"):
    """A list of (W, b) pairs, W of shape (inputs, outputs) as in the template.

    sizes = [784, 100, 10] gives two layers: 784 -> 100 -> 10.  The weights are
    drawn with the variance of Eq. (8.74) (He) for ReLU and 1/inputs otherwise,
    the biases are set to 0.01 as in the template.
    """
    rng = np.random.default_rng(2026) if rng is None else rng
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        scale = np.sqrt(2.0 / n_in) if hidden_activation == "relu" else np.sqrt(1.0 / n_in)
        W = rng.normal(0.0, scale, (n_in, n_out))
        b = np.zeros(n_out) + 0.01
        layers.append((W, b))
    return layers


JAX_ACTIVATIONS = {"tanh": jnp.tanh, "sigmoid": jax.nn.sigmoid, "relu": jax.nn.relu,
                   "leaky_relu": jax.nn.leaky_relu, "elu": jax.nn.elu, "gelu": jax.nn.gelu,
                   "swish": jax.nn.swish, "softplus": jax.nn.softplus}


def network(layers, X, activation="tanh"):
    """The feed-forward pass of week 41 (Eq. 8.21) in jax.numpy, linear output; returns N(x) as a vector.

    `layers` is the list of (W, b) pairs of create_layers; X has the inputs in
    rows (one column for an ODE, two for a PDE in x and t).
    """
    f = JAX_ACTIVATIONS[activation]
    a = X
    for l, (W, b) in enumerate(layers):
        z = a @ W + b
        a = f(z) if l < len(layers) - 1 else z      # linear output: the "mse" head of week 41
    return a[:, 0]


def to_jax(layers):
    """numpy (W, b) pairs from create_layers -> jax arrays (so that jax.grad can trace through them)."""
    return [(jnp.asarray(W), jnp.asarray(b)) for W, b in layers]

