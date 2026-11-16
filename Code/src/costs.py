"""
The two ways of imposing the initial and boundary conditions.

- Hard-coded: a trial function that satisfies the conditions exactly (a known
  function plus a factor that vanishes on them times the network), and its cost,
  the mean squared residual.
- Soft (PINN): the network is the solution, and the cost is the mean squared
  residual plus weighted squared violations of the conditions. Optional data
  term and trainable sigma for the volatility extension.
"""
