# Quick start

```@meta
CurrentModule = PEPit
```

This page gives the standard `PEPit.jl` workflow for a worst-case analysis. The
example computes a guarantee for gradient descent on the class of `L`-smooth
convex functions. The code blocks below are *executed* when this documentation
is built, so the values shown are produced by `PEPit.jl` itself.

## Installation

From the Julia package manager, add the package and instantiate its
dependencies:

```julia
] add https://github.com/PerformanceEstimation/PEPit.jl
```

During development from this repository, use the package project:

```bash
julia --project=PEPit.jl
```

## A first PEP

Load the package, an ordered-dictionary type for the class parameters, and a
conic solver:

```@example pep
using PEPit
using OrderedCollections
using Clarabel
```

Create a PEP and declare the function class (parameters are passed through an
`OrderedDict`):

```@example pep
problem = PEP()
param = OrderedDict("L" => 3.0)
func = declare_function!(problem, SmoothConvexFunction, param)
nothing # hide
```

Declare a stationary point ``x_\star`` (where the gradient vanishes), its
function value ``f_\star``, an initial point ``x_0``, and the initial condition
``\|x_0 - x_\star\|^2 \le 1``:

```@example pep
xs = stationary_point!(func)
fs = value!(func, xs)
x0 = set_initial_point!(problem)

set_initial_condition!(problem, (x0 - xs)^2 <= 1)
nothing # hide
```

Describe the algorithm symbolically. Here `gradient!(func, x)` creates or reuses
an oracle evaluation according to the function class and its `reuse_gradient`
setting. (The loop assigns to the module-level `x`, hence the `global`.)

```@example pep
L = 3.0
gamma = 1 / L
n = 4

x = x0
for _ in 1:n
    global x = x - gamma * gradient!(func, x)
end
```

Set the performance metric ``f(x_n) - f_\star`` and solve the SDP:

```@example pep
set_performance_metric!(problem, value!(func, x) - fs)

pepit_tau = solve!(problem; solver = Clarabel.Optimizer, verbose = false)
```

The returned value is the worst-case constant ``\tau`` in the guarantee

```math
f(x_n) - f_\star \;\leq\; \tau \, \|x_0 - x_\star\|^2 .
```

## Recovering the worst-case instance

After [`solve!`](@ref), [`evaluate`](@ref) recovers one numerical realization of
any symbolic [`Point`](@ref) or [`Expression`](@ref) from the solved Gram
matrix. This is how the worst-case iterates and function values are
reconstructed:

```@example pep
evaluate(x)                       # coordinates of the last iterate x_n
```

```@example pep
evaluate(value!(func, x) - fs)    # the attained value f(x_n) - f_star
```

## Explicit dual certificate

The dual multipliers of the SDP form a machine-checkable proof of the bound.
[`solve_dual!`](@ref) builds the primal SDP, dualizes it with `Dualization.jl`,
solves the dual, and returns a [`DualPEPCertificate`](@ref):

```@example pep
certificate = solve_dual!(problem; verbose = false)
certificate.dual_value            # dual objective: matches the primal tau
```

The certificate exposes the multiplier blocks — `α` for the performance metric,
`λ`/`ν` for inequality/equality conditions, `θ` for the interpolation
constraints, and `S`/`Y` for the PSD blocks. After solving, [`eval_dual`](@ref)
returns the dual value attached to an individual [`Constraint`](@ref) or
[`PSDMatrix`](@ref).

## Dimension-reduction heuristics

Worst-case Gram matrices are often low rank, so a low-dimensional worst-case
example usually exists. [`solve!`](@ref) can post-process the solution with a
trace-minimization step (`tracetrick`) or several log-det iterations
(`logdetiters`) while preserving the optimal value:

```@example pep
solve!(problem; verbose = false, tracetrick = true)
```

Relevant keywords of [`solve!`](@ref): `tracetrick::Bool`, `logdetiters::Int`,
`eig_regularization`, and `tol_dimension_reduction` (the objective degradation
tolerated during reduction).
