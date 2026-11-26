# Depth study of the kink-free trial network: 1 to 8 layers of 50 nodes, learning rate 1e-2, five seeds.
# Saves the results; depth_plot.py makes the figure. (The architecture scan stopped at 5 layers; this checks
# what happens beyond, and how much the result varies between seeds.)
import numpy as np

from src.training import grid_search, RESULTS_DIR

layers = [1, 2, 3, 4, 5, 6, 7, 8]
nodes = [50]
learning_rates = [1e-2]
n_seeds = 5

activation = "gelu"         # chosen in the activation scan, so the depth study matches the rest of part d

validation, train_time = grid_search("kink_free", layers, nodes, learning_rates, n_seeds, activation=activation)
np.savez(RESULTS_DIR / "depth_scan_kink_free.npz", validation=validation[:, 0, 0, :], train_time=train_time[:, 0, 0, :],
         layers=layers, nodes=nodes, learning_rate=learning_rates[0], activation=activation)
