# Project 3, FYS-STK3155/FYS4155: option pricing as a heat equation

**Authors:**
 - Albert Sjåvåg
 - Teodor Aursnes

 (University of Oslo, autumn 2026)

Path A of project 3. We solve the Black–Scholes equation for a European call option,
transformed to the one-dimensional heat equation $u_\tau = u_{xx}$ with a kink in the
initial condition, with three methods: an explicit finite-difference scheme, a neural
network with hard-coded initial and boundary conditions, and a physics-informed neural
network (PINN). All three are compared with the Black–Scholes formula, and the test case
of the project text, $u(x, 0) = \sin(\pi x)$, is used to verify the code. We study how the
architecture, the activation functions, the collocation points and the training affect
the networks, and finally recover the volatility from noisy prices with a PINN.

The report is in `report/` (not written yet).

## Installation

```bash
pip install -r requirements.txt
```

## Structure

Everything is run from the `Code/` folder.

```
Code/src/          shared modules
Code/parts/        one script per analysis, part a to d and the volatility extension
Code/tests/        unit tests (pytest)
Code/figures/      all figures in the report (PNG)
Code/outputs/      the printed output of every script
Code/results/      trained parameters and scan results, so figures can be redrawn without training
Code/run_all.ipynb runs every script and the tests, and shows the figures
LLM_usage.md       declaration of LLM use, file by file
```

| Module | Content |
|---|---|
| `src/problems.py` | The common seed, the two problems in heat-equation form, the Black–Scholes formula, the change of variables, and synthetic prices with noise |
| `src/finite_difference.py` | The explicit forward-Euler scheme |
| `src/network.py` | The feed-forward network in JAX, and $u_t$, $u_{xx}$ and the residual by automatic differentiation |
| `src/costs.py` | The trial function with hard-coded conditions and its cost, and the PINN cost function |
| `src/training.py` | Collocation and validation points, and the training loop |
| `src/optimiser.py` | Gradient descent, momentum, AdaGrad, RMSprop and Adam (from projects 1 and 2) |
| `src/plotting.py` | Common figure settings and `save_fig` (from project 2) |

## Running

**Everything at once:** run all cells in `Code/run_all.ipynb`. All random numbers use the seed 2026.

**One script at a time:** from `Code/`, run a script as a module and save its output, for example

```bash
cd Code
python -m parts.part_b.convergence | tee outputs/parts.part_b.convergence.txt
```

| Script | What it does | Figure |
|---|---|---|
| **Part a: the problem** | | |
| `parts/part_a/exact_solution.py` | The Black–Scholes value for several times to expiry | `exact_solution.png` |
| **Part b: the explicit scheme** | | |
| `parts/part_b/profiles.py` | Absolute error against $x$ at $t_1$ and $t_2$ for $\Delta x = 1/10$ and $1/100$ | `fd_profiles.png` |
| `parts/part_b/convergence.py` | Maximal error against $\Delta x$, and the scheme above the stability limit | `fd_convergence.png` |
| **Part c: the networks** | | |
| `parts/part_c/train.py` | Trains both networks over several seeds and saves them in `results/` | `training_curves.png` |
| `parts/part_c/compare.py` | All methods against the exact solution over the domain and at $t_1$ and $t_2$ | `error_surface.png`, `methods_t1_t2.png` |
| **Part d: network complexity** | | |
| `parts/part_d/architecture.py` | Number of layers and nodes for each activation function | `architecture.png` |
| `parts/part_d/collocation.py` | Number and placement of the collocation points | `collocation.png` |
| `parts/part_d/loss_weights.py` | Weights of the PINN cost function | `loss_weights.png` |
| `parts/part_d/optimizer.py` | Optimizer and learning rate | `optimizer.png` |
| `parts/part_d/final_test.py` | The chosen configurations on the test grid, and error against computing time | `cost.png` |
| **Extension: implied volatility** | | |
| `parts/volatility/implied_vol.py` | Volatility from noisy prices: PINN against root finding | `implied_vol.png` |

The figures are planned and not made yet.

## Seeds

All random numbers derive from `SEED = 2026` in `src/problems.py`. Every network result is
repeated over several seeds, with separate seeds for the initial weights and for the
collocation points, and reported as mean and standard deviation. The explicit scheme has no
randomness.

## Tests

```bash
cd Code
python -m pytest -v
```

`tests/test_implementation.py` checks the exact solutions, the convergence and stability of
the explicit scheme, and the automatic-differentiation derivatives and trial functions.

## References

- F. Black and M. Scholes, *The Pricing of Options and Corporate Liabilities*, Journal of
  Political Economy **81**, 637 (1973).
- I. E. Lagaris, A. Likas and D. I. Fotiadis, *Artificial Neural Networks for Solving
  Ordinary and Partial Differential Equations*, IEEE Transactions on Neural Networks **9**,
  987 (1998).
- M. Raissi, P. Perdikaris and G. E. Karniadakis, *Physics-informed neural networks*,
  Journal of Computational Physics **378**, 686 (2019).
- M. Hjorth-Jensen, lecture notes and weekly material for FYS-STK3155/FYS4155.

The use of LLMs is declared in `LLM_usage.md`.
