import numpy as np

from src.training import train_config, RESULTS_DIR

methods = ["kink_free", "pinn"]      # the naive net is a control in final_test:its validation cost is blind to the kink
n_cols = [250, 1000, 4000]
placements = ["uniform", "kink"]     # "kink": half of the points near the kink, |x| < 0.2 and the first tenth of the time
n_seeds = 3

for method in methods:
    # Everything else from the earlier scans: tune one thing at a time.
    arch = np.load(RESULTS_DIR / f"architecture_scan_{method}.npz")
    a, b, c = np.unravel_index(np.argmin(arch["validation"].mean(axis=-1)), arch["validation"].shape[:3])
    sizes = (2,) + (int(arch["nodes"][b]),) * int(arch["layers"][a]) + (1,)
    learning_rate = float(arch["learning_rates"][c])
    act = np.load(RESULTS_DIR / f"activation_scan_{method}.npz")
    mean = act["validation"].mean(axis=1)
    mean[list(act["activations"]).index("relu")] = np.inf     # ReLU is ruled out by theory, not by its cost
    activation = str(act["activations"][np.argmin(mean)])
    w = 100.0                                                 # PINN weight; not used by the kink-free net
    if method == "pinn":
        lw = np.load(RESULTS_DIR / "loss_weight_scan.npz")
        w = float(lw["weights"][np.argmin(lw["terms"].sum(axis=2).mean(axis=1))])

    validation = np.zeros((len(n_cols), len(placements), n_seeds))
    train_time = np.zeros_like(validation)
    for j, n_col in enumerate(n_cols):
        for p, placement in enumerate(placements):
            for i in range(n_seeds):
                run = train_config(method, seed_offset=i, sizes=sizes, learning_rate=learning_rate,
                                   activation=activation, n_col=n_col, placement=placement,
                                   w_initial=w, w_boundary=w)
                validation[j, p, i] = run["validation"]
                train_time[j, p, i] = run["timing"]["train"]

    np.savez(RESULTS_DIR / f"collocation_scan_{method}.npz", validation=validation, train_time=train_time,
             n_cols=n_cols, placements=placements, sizes=sizes, learning_rate=learning_rate,
             activation=activation, w=w)
