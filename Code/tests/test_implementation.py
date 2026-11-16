"""
Tests of the implementation:
- the exact solutions satisfy the equation and the conditions, the
  Black-Scholes formula matches a known price, and the change of variables goes
  back and forth;
- the explicit scheme converges at the expected rate on the sin(pi x) case and
  blows up above dt/dx^2 = 1/2;
- the automatic-differentiation derivatives match a function with known
  derivatives, and the trial functions satisfy the conditions exactly.
"""
