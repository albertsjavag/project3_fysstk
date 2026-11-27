import numpy as np
import matplotlib.pyplot as plt

from src.problems import L, alpha, sigma, bs_initial
from src.problems import bs_left_boundary, bs_right_boundary, bs_exact, sin_exact, sin_initial
from src.finite_difference import explicit_scheme
from src.plotting import save_fig

# dx = 1/10 to 1/100, the range the task asks about. dx = 0.1 / n with whole n, so that one week (one step on the
# coarsest grid) is a whole number of steps, n^2, on every grid, and the strike x = 0 is a grid point.
dxs = [0.1 / n for n in range(1, 11)]
weeks = [1, 4, 13, 52]
taus = [sigma**2 / 2 * w / 52 for w in weeks]
zero = lambda tau: 0.0

errors_bs = np.zeros((len(dxs), len(weeks)))
errors_sin = np.zeros(len(dxs))

for i, dx in enumerate(dxs):
    x, u_fd = explicit_scheme(bs_initial, bs_left_boundary, bs_right_boundary, -L, L, dx, alpha, taus)
    for j, tau in enumerate(taus):
        errors_bs[i, j] = np.abs(u_fd[j] - bs_exact(x, tau)).max()
    x, u_fd = explicit_scheme(sin_initial, zero, zero, 0, 1, dx, alpha, [0.02])
    errors_sin[i] = np.abs(u_fd[0] - sin_exact(x, 0.02)).max()

print("max |u_fd - u_exact|, rows dx, columns: sin (tau = 0.02), Black-Scholes " + ", ".join(f"{w} wk" for w in weeks))
for i, dx in enumerate(dxs):
    print(f"dx = 1/{round(1 / dx):<4d}" + "".join(f"{e:10.1e}" for e in [errors_sin[i], *errors_bs[i]]))

log_dx = np.log(dxs)
print(f"sin: slope {np.polyfit(log_dx, np.log(errors_sin), 1)[0]:.2f}")
for j, w in enumerate(weeks):
    local_slopes = np.diff(np.log(errors_bs[:, j])) / np.diff(log_dx)
    print(f"{w:2d} weeks: local slopes " + " ".join(f"{s:4.1f}" for s in local_slopes)
          + f", kink width sqrt(2 tau) = {np.sqrt(2 * taus[j]):.3f}")

fig, ax = plt.subplots()
for j, w in enumerate(weeks):
    line, = ax.loglog(dxs, errors_bs[:, j], "o-", label=f"{w} week" + ("s" if w > 1 else ""))
    # The width sqrt(2 tau) of the smoothed kink: the error falls as dx^2 once dx is below it. Drawn only when it
    # lies inside the plotted range (1 and 4 weeks; 13 weeks is at the edge and 52 weeks beyond it).
    kink_width = np.sqrt(2 * taus[j])
    if kink_width < 0.9 * dxs[0]:
        ax.axvline(kink_width, color=line.get_color(), ls="--", lw=0.8)
ax.loglog(dxs, errors_sin, "s--", color="gray", label="sin")
ax.loglog(dxs, 0.3 * errors_sin[0] * (np.array(dxs) / dxs[0])**2, "k:", label="slope 2")   # below sin, not on top
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"max $|u_{fd} - u_{exact}|$")
ax.legend(fontsize=7, loc="lower right")

save_fig(fig, "fd_convergence")
