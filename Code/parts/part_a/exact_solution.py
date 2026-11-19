"""
Part a: the exact solution. The Black-Scholes value against stock price for
several times to expiry, showing how the kink of the payoff is smoothed out.
Figure: exact_solution.png.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.problems import black_scholes_call, K, T, L
from src.plotting import save_fig

# The cut-off domain x in [-L, L] in stock prices. The figure shows only the part around the
# strike, where the kink is; far from it the price is close to 0 or to a straight line.
S_domain = (K * np.exp(-L), K * np.exp(L))
S = np.linspace(60, 160, 401)
time_to_expiry = np.linspace(0, T, 101)
S_grid, time_grid = np.meshgrid(S, time_to_expiry)
C_grid = black_scholes_call(S_grid, time_grid)   # the function takes the whole grid at once

print(f"Strike K = {K:g}, expiry T = {T:g} year, domain S in [{S_domain[0]:.1f}, {S_domain[1]:.1f}]")
print(f"Price at S = K today (T - t = {T:g}): {float(black_scholes_call(K, T)):.4f}")

# Constrained layout and the tight bounding box both cut the 3D axis labels, so the margins are set by hand.
fig = plt.figure(figsize=(3.4, 3.0), layout="none")
ax = fig.add_subplot(projection="3d")
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=1.0)
ax.plot_surface(S_grid, time_grid, C_grid, cmap="viridis", linewidth=0, rcount=60, ccount=60, alpha=0.9)

# The payoff (T - t = 0, kinked) and one year before expiry (smooth) drawn on the surface.
for years in (0.0, 1.0):
    ax.plot(S, np.full_like(S, years), black_scholes_call(S, years), color="red", lw=1.2, zorder=10)

ax.set_xlabel("Stock price $S$", labelpad=-1)
ax.set_ylabel("$T-t$ (years)", labelpad=-1)
ax.set_zlabel("Call price $C$", labelpad=-3)
ax.set_xticks([60, 100, 140])
ax.set_yticks([0, 0.5, 1.0])
ax.set_zticks([0, 20, 40, 60])
ax.tick_params(pad=0)
ax.view_init(elev=18, azim=-115)   # seen from the expiry side, so the kink is in front

with plt.rc_context({"savefig.bbox": "standard"}):
    save_fig(fig, "exact_solution")
