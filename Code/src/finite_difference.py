import numpy as np

def explicit_scheme(u_initial, left_boundary, right_boundary, x_min, x_max, dx, taus, alpha=0.5):
    n = round((x_max - x_min) / dx)
    x = np.linspace(x_min, x_max, n + 1)
    u = np.array(u_initial(x), dtype=float)   # numpy copy, so we can update it in place
    tau = 0.0
    u_saved = []

    for tau_end in taus:
        # Largest stable step is alpha dx^2; take a few more, slightly smaller steps so we land on tau_end exactly.
        n_steps = int(np.ceil((tau_end - tau) / (alpha * dx**2)))
        dtau = (tau_end - tau) / n_steps
        for _ in range(n_steps):
            u[1:-1] = u[1:-1] + dtau / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
            tau += dtau
            u[0] = left_boundary(tau)
            u[-1] = right_boundary(tau)
        tau = tau_end   # remove the round-off from summing the steps
        u_saved.append(u.copy())

    return x, np.array(u_saved)
