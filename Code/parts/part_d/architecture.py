# Grid search over layers x nodes x learning rate, adapted from the example report's code (D. Haas,
# FYS-STK4155 project 3, GridSearch.py). Saves the results; architecture_plot.py makes the figure.
import numpy as np

from src.training import grid_search, RESULTS_DIR

methods = ["pinn"]                 # the kink-free scan is done; run one method at a time
n_seeds = 3
layers = [1, 2, 3, 4, 5]
nodes = [10, 30, 50]
learning_rates = [3e-3, 1e-2, 3e-2, 1e-1]

for method in methods:
    validation, train_time = grid_search(method, layers, nodes, learning_rates, n_seeds)
    np.savez(RESULTS_DIR / f"architecture_scan_{method}.npz", validation=validation, train_time=train_time,
             layers=layers, nodes=nodes, learning_rates=learning_rates)
