import numpy as np


def explicit_scheme(u_initial, left_boundary, right_boundary, x_min, x_max, dx, alpha, taus):
    if alpha > 0.5:
        raise ValueError(f"alpha = {alpha} is above the stability limit 1/2")
    n = round((x_max - x_min) / dx)
    x = np.linspace(x_min, x_max, n + 1)
    u = np.array(u_initial(x), dtype=float)   # numpy copy, so we can update it in place

    # One fixed time step for the whole run, so the answer at a time does not depend on which other times we ask for.
    dtau = alpha * dx**2
    target_steps = [round(tau / dtau) for tau in taus]
    for tau, m in zip(taus, target_steps):
        if abs(m * dtau - tau) > 1e-9 * tau:
            raise ValueError(f"tau = {tau} is not a whole number of steps of dtau = {dtau}")

    step = 0
    u_saved = []
    for m in target_steps:
        while step < m:
            u[1:-1] = u[1:-1] + alpha * (u[2:] - 2 * u[1:-1] + u[:-2])
            step += 1
            u[0] = left_boundary(step * dtau)
            u[-1] = right_boundary(step * dtau)
        u_saved.append(u.copy())

    return x, np.array(u_saved)
