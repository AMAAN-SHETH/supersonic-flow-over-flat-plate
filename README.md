# Supersonic Flow Over a Flat Plate Solver

This solver simulates **supersonic flow over a flat plate at zero angle of attack**, based on the formulation in *John D. Anderson’s "Computational Fluid Dynamics: The Basics with Applications"*, Chapter 10. It solves the full Navier–Stokes equations using the **MacCormack predictor–corrector scheme** and finite-difference spatial discretization.

---

## Theoretical Background

According to Anderson (Chapter 10), this canonical problem demonstrates boundary layer behavior in supersonic flow and serves as a benchmark for viscous-flow solvers. A thin laminar boundary layer develops over the sharp leading edge, and accurately capturing the steep gradients near the wall demands careful numerical treatment.

---

## Numerical Methodology

- **MacCormack Scheme**: A two-step explicit time-marching predictor–corrector approach applied to the unsteady form of the Navier–Stokes equations to achieve a steady-state solution.
- **Finite Difference Discretization**: Spatial derivatives for convective and viscous terms are computed using central differences to achieve second-order accuracy.
- **Boundary Conditions**: Freestream conditions at the inlet; no‑slip, constant wall (or adiabatic) conditions at the plate surface; symmetry or outflow conditions elsewhere.
- **Grid Layout**: Structured Cartesian grid, refined near the wall to resolve the viscous sublayer and pressure jump at the leading edge.

---

