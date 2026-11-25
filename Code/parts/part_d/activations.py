
from src.training import train_config, RESULTS_DIR

import numpy as np

methods = ["kink_free", "pinn"]
activations = ["tanh", "sigmoid", "gelu", "swish", "softplus", "relu"]
n_seeds = 3

for method in methods:
    # The best architecture and learning rate from the architecture scan: tune one thing at a time.
    scan = np.load(RESULTS_DIR / f"architecture_scan_{method}.npz")
    a, b, c = np.unravel_index(np.argmin(scan["validation"].mean(axis=-1)), scan["validation"].shape[:3])
    sizes = (2,) + (int(scan["nodes"][b]),) * int(scan["layers"][a]) + (1,)
    learning_rate = float(scan["learning_rates"][c])

    validation = np.zeros((len(activations), n_seeds))
    train_time = np.zeros_like(validation)
    for j, activation in enumerate(activations):
        for i in range(n_seeds):
            run = train_config(method, seed_offset=i, sizes=sizes, learning_rate=learning_rate, activation=activation)
            validation[j, i] = run["validation"]
            train_time[j, i] = run["timing"]["train"]

    np.savez(RESULTS_DIR / f"activation_scan_{method}.npz", validation=validation, train_time=train_time,
                activations=activations, sizes=sizes, learning_rate=learning_rate)