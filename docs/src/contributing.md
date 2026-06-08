# Contributing

`PEPit.jl` is research software. Contributions should preserve mathematical
correctness, reproducibility, and parity with the intended PEP formulation. The
most common contributions are new **function/operator classes**, new
**primitive steps** (black-box oracles), and new **worked examples**.

## Local development

Instantiate the package project and run the test suite (it uses the open-source
`Clarabel` solver, so no commercial license is required):

```bash
julia --project=PEPit.jl -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Build the documentation:

```bash
julia --project=PEPit.jl/docs -e 'using Pkg; Pkg.develop(PackageSpec(path="PEPit.jl")); Pkg.instantiate()'
julia --project=PEPit.jl/docs PEPit.jl/docs/make.jl
```

## General guidelines

- Follow standard Julia conventions: `CamelCase` for types, a trailing `!` for
  mutating functions, and parameters passed through an `OrderedDict`.
- Document every exported symbol with a docstring attached immediately above it.
  Use `@doc raw"""..."""` whenever the docstring contains LaTeX, and write
  display math inside a ` ```math ` fence (keep the whole expression — including
  any `\left\{ ... \right.` — inside a single fence).
- Add a test for every new class, step, or example; keep examples deterministic
  and solver-light.
- Update the documentation (the relevant `docs/src/api/*.md` page) and add a line
  to `CHANGELOG.md` describing the change.

## Adding a function or operator class

Function classes live in `src/functions/`, operator classes in `src/operators/`.
A class is a small `struct` that wraps a [`PEPFunction`](@ref) and overrides
`add_class_constraints!` to add its interpolation (or operator) inequalities.
Using `ConvexFunction` (`src/functions/convex_function.jl`) as a template:

```julia
@doc raw"""
    MyFunction(param; reuse_gradient=false)

One-line summary, then the interpolation conditions in a `math` fence, and a
clickable reference to the paper that introduces them.
"""
mutable struct MyFunction <: AbstractFunction
    mu::Float64                 # class parameters, if any
    L::Float64
    _PEPit_func::PEPFunction    # required: stores oracle calls and constraints

    function MyFunction(param=OrderedDict(); is_leaf=true,
                        decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(; is_leaf, decomposition_dict, reuse_gradient)
        return new(param["mu"], param["L"], func)
    end
end

# Delegate the oracle/utility methods to the wrapped PEPFunction:
gradient!(f::MyFunction, p::Point)             = gradient!(f._PEPit_func, p)
value!(f::MyFunction, p::Point)                = value!(f._PEPit_func, p)
stationary_point!(f::MyFunction)              = stationary_point!(f._PEPit_func)
add_constraint!(f::MyFunction, c::Constraint) = add_constraint!(f._PEPit_func, c)
_get_pep_func(f::MyFunction)                  = f._PEPit_func

# The mathematical content: add one inequality per ordered pair of oracle points.
function add_class_constraints!(f::MyFunction)
    pts = f._PEPit_func.list_of_points
    for pi in pts, pj in pts
        pi == pj && continue
        xi, gi, fi = pi
        xj, gj, fj = pj
        add_constraint!(f, fi - fj >= gj * (xi - xj))  # replace with your class' inequality
    end
end
```

Then: (1) `include` the file in `src/PEPit.jl`, (2) add the type to the `export`
list, and (3) list it in `docs/src/api/functions.md` (or `operators.md`).

## Adding a primitive step

Primitive steps live in `src/primitive_steps/`. A step is a `!`-suffixed function
that creates symbolic points/gradients/values, attaches the appropriate
constraints with `add_constraint!`, and returns the new symbolic objects.
Document it with `@doc raw"""..."""`, including the defining display-math relation
and a paper reference (see `src/primitive_steps/inexact_gradient_step.jl`). Export
it and list it in `docs/src/api/steps.md`.

## Adding an example

Examples live in `examples/<family>/`. Each file defines a single
`wc_<method_name>(...)` function whose `@doc raw"""..."""` docstring is extracted
verbatim into the documentation, so it must sit immediately above the
`function wc_...` line. Follow this structure (see
`examples/unconstrained_convex_minimization/gradient_descent.jl`):

```julia
@doc raw"""
    wc_my_method(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

## Problem statement
Define the function class and assumptions.

## Algorithm
The iteration, in a `math` fence.

## Performance metric and initial condition
The quantity being bounded and the normalization, e.g.
``f(x_n) - f_\star \le \tau(L, \gamma, n)\, \|x_0 - x_\star\|^2``.

## Theoretical guarantee
State whether the bound is **tight**, an **upper**, or a **lower** bound, with
the closed-form ``\tau``.

## References
Clickable reference(s) to the introducing paper(s).

# Arguments
- `L`: smoothness parameter.
- `gamma`: step size.
- `n`: number of iterations.

# Returns
- `pepit_tau`: the worst-case value computed by `PEPit.jl`.
- `theoretical_tau`: the reference value from the literature.
"""
function wc_my_method(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    # build the PEP, solve it, and return (pepit_tau, theoretical_tau)
end
```

The Examples overview and per-example pages are then generated automatically by
`docs/make.jl` from this docstring — no manual edit of `examples.md` is needed.

## Writing the test

Add a `@testset` that runs the example and compares the computed value to the
theoretical one within a relative tolerance:

```julia
@testset "my_method" begin
    pepit_tau, theoretical_tau = wc_my_method(1.0, 1.0, 5; verbose=false)
    @test isapprox(pepit_tau, theoretical_tau; rtol=1e-4)     # if the bound is tight
    # or, for a (non-tight) upper bound:
    # @test pepit_tau <= theoretical_tau * (1 + 1e-4)
end
```

## Documentation style

- Prefer mathematically precise prose with Julia-specific syntax.
- Keep public docstrings concise and attach them immediately above the object.
- Use examples that are deterministic and solver-light.
- Link long-running examples instead of executing them in the documentation
  build.
