import numpy as np

from src.training import train_config, RESULTS_DIR

# Kink-free trial network only (the PINN uses Adam, as in the lectures). Each optimizer gets its own learning
# rates: plain gradient descent and momentum diverge in the first step already at 1e-4, because the residual
# carries the factor 1 / tau_max = 50, so their gradients are huge.
learning_rates = {
    "plain": [1e-6, 1e-5, 1e-4],
    "momentum": [1e-6, 1e-5, 1e-4],
    "rmsprop": [1e-3, 1e-2, 1e-1],
    "adam": [1e-3, 1e-2, 1e-1],
}
n_seeds = 3

# Everything else from the earlier scans: tune one thing at a time.
arch = np.load(RESULTS_DIR / "architecture_scan_kink_free.npz")
a, b, c = np.unravel_index(np.argmin(arch["validation"].mean(axis=-1)), arch["validation"].shape[:3])
sizes = (2,) + (int(arch["nodes"][b]),) * int(arch["layers"][a]) + (1,)
act = np.load(RESULTS_DIR / "activation_scan_kink_free.npz")
mean = act["validation"].mean(axis=1)
mean[list(act["activations"]).index("relu")] = np.inf     # ReLU is ruled out by theory, not by its cost
activation = str(act["activations"][np.argmin(mean)])

optimizers = list(learning_rates)
validation = np.zeros((len(optimizers), 3, n_seeds))
train_time = np.zeros_like(validation)
for o, optimizer in enumerate(optimizers):
    for j, learning_rate in enumerate(learning_rates[optimizer]):
        for i in range(n_seeds):
            run = train_config("kink_free", seed_offset=i, sizes=sizes, activation=activation,
                               learning_rate=learning_rate, optimizer=optimizer)
            validation[o, j, i] = run["validation"]
            train_time[o, j, i] = run["timing"]["train"]

np.savez(RESULTS_DIR / "optimizer_scan_kink_free.npz", validation=validation, train_time=train_time,
         optimizers=optimizers, learning_rates=np.array([learning_rates[o] for o in optimizers]),
         sizes=sizes, activation=activation)

# One line per optimizer, at its best learning rate.
print(f"kink-free trial network {sizes}, {activation}, {n_seeds} seeds, 6000 steps")
for o, optimizer in enumerate(optimizers):
    j = np.argmin(validation[o].mean(axis=1))
    print(f"{optimizer:9s} best learning rate {learning_rates[optimizer][j]:.0e}: validation "
          f"{validation[o, j].mean():.2e} ± {validation[o, j].std():.0e}, training {train_time[o, j].mean():.1f} s")
