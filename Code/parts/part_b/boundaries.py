import numpy as np

from src.problems import K, k, alpha, t1, t2, to_heat_variables, u_to_price, bs_initial, bs_exact
from src.finite_difference import explicit_scheme

# The two right boundary conditions and three cut-offs L, on the fine grid dx = 1/100, so the discretisation error
# is small and the boundary error shows. (a) C = S, (b) C = S - K exp(-r (T - t)), both as u at x = L.
taus = [to_heat_variables(K, t)[1] for t in (t1, t2)]
labels = ["1 week", "1 year"]

for L in (0.5, 1.0, 1.5):
    for kind in ("a", "b"):
        def right_boundary(tau):
            stock = np.exp((k + 1) * L / 2 + (k + 1)**2 * tau / 4)
            return stock if kind == "a" else stock - np.exp((k - 1) * L / 2 + (k - 1)**2 * tau / 4)

        x, u_fd = explicit_scheme(bs_initial, lambda tau: 0.0, right_boundary, -L, L, 0.01, alpha, taus)
        row = []
        for j, tau in enumerate(taus):
            C_error = np.abs(u_to_price(u_fd[j], x, tau) - u_to_price(bs_exact(x, tau), x, tau))
            row.append(f"{labels[j]}: max price error {C_error.max():6.3f} at S = {K * np.exp(x[C_error.argmax()]):3.0f}")
        print(f"L = {L}, boundary ({kind}):   " + "   ".join(row))
