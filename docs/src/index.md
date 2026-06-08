# PEPit.jl

`PEPit.jl` is a Julia implementation of Performance Estimation Programming
for computer-assisted worst-case analysis of first-order algorithms. A user
describes an algorithm in symbolic Julia code, declares the class of functions
or operators under study, specifies initial conditions and performance
metrics, and asks `PEPit.jl` to build and solve the resulting semidefinite
program through JuMP-compatible solvers.

The package follows the mathematical workflow of Python `PEPit`, but the
documentation here describes the Julia API and implementation. In particular,
points, gradients, and iterates are represented by [`Point`](@ref) objects;
function values and inner products are represented by [`Expression`](@ref)
objects; and [`solve!`](@ref) turns those symbolic relations into a JuMP SDP
with a Gram matrix variable.

## Documentation map

- [Quick start](@ref): build and solve a first PEP in Julia.
- [Core workflow](@ref): main PEP objects and solver entry points.
- [Function classes](@ref): interpolation models for convex, smooth, strongly
  convex, Lipschitz, and related function classes.
- [Operator classes](@ref): monotone, cocoercive, Lipschitz, nonexpansive,
  and linear operator models.
- [Primitive steps](@ref): reusable symbolic building blocks for algorithm
  descriptions.
- [Examples](@ref): categorized links to Julia example scripts.
- [Tutorials](@ref): literate Julia, notebook, PDF, and Pluto tutorial assets.

## Solvers

The default solver in [`solve!`](@ref) is `Clarabel.Optimizer`. The package also
supports other JuMP-compatible conic solvers, including Mosek when a license is
available. Solver choice is passed as a keyword argument:

```julia
tau = solve!(problem; solver = Clarabel.Optimizer, verbose = false)
```

## Explicit dual certificates

Use [`solve_dual!`](@ref) to build the primal SDP, dualize it with
`Dualization.jl`, and return a [`DualPEPCertificate`](@ref). The certificate
stores multipliers for performance metrics, initial conditions, class
constraints, and PSD blocks.
