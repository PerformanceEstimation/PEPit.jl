# PEPit.jl

[![Documentation](https://img.shields.io/badge/Documentation-stable-purple.svg)](https://PerformanceEstimation.github.io/PEPit.jl/stable/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v0.1.2-blue.svg)](https://github.com/PerformanceEstimation/PEPit.jl/releases/tag/v0.1.2)

`PEPit.jl` is a native Julia implementation of the Performance Estimation Programming (PEP) methodology [1,2,3] and the Python package `PEPit` [4] for worst-case analysis of first-order optimization algorithms. The core idea in PEP is to model the design and analysis of first-order optimization algorithms as higher-level optimization problems called performance estimation problems (PEPs), which are semidefinite programs (SDPs). We then solve these SDPs numerically to obtain tight worst-case bounds for known algorithms and also to discover new algorithms under suitable conditions.

The intent of this Julia package is to be functionally equivalent to existing packages such as `PESTO` [5] and `PEPit` while providing a clean, Julia-native API along with a broader support of commercial and open-source solvers under the `JuMP` ecosystem [6].

## Install and test

You can install the package by typing the following in the Julia REPL:

```julia
] add PEPit
```

To install directly from the GitHub repo, type the following in the Julia REPL:
```julia
] add https://github.com/PerformanceEstimation/PEPit.jl
```

Then in Julia, you can run the following test to see if the package is working as intended:

```julia
] test PEPit
```

## Reproducible [`Pluto`](`Pluto`) notebook to try out `PEPit.jl`

A completely open-source reproducible `Pluto` notebook to try out a simple `PEPit.jl` example is available at:

[https://pluto.land/n/ds42k2rm](https://pluto.land/n/ds42k2rm)

From this link, you can either download the `Pluto` notebook file and run it locally with `Pluto`, or run the notebook with [`Binder`](https://plutojl.org/en/docs/binder/) directly in your browser!

## Minimal example

Below is a condensed example following the style of the examples in `examples/`; for other examples please take a look in the `examples/` folder. It computes a worst-case bound  for accelerated gradient method for smooth strongly convex minimization (please also see the `Step-by-Step tutorial` for more details):

```julia
using PEPit, OrderedCollections

function wc_accelerated_gradient_strongly_convex(mu, L, n; verbose=false)
    problem = PEP()
    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param; reuse_gradient=true)

    xs = stationary_point!(func)
    fs = value!(func, xs)
    x0 = set_initial_point!(problem)

    set_initial_condition!(problem, value!(func, x0) - fs + mu / 2 * (x0 - xs)^2 <= 1)

    kappa = mu / L
    x_new, y = x0, x0
    for i in 1:n
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        y = x_new + (1 - sqrt(kappa)) / (1 + sqrt(kappa)) * (x_new - x_old)
    end

    set_performance_metric!(problem, value!(func, x_new) - fs)

    PEPit_tau = solve!(problem, verbose=verbose)
    theoretical_tau = (1 - sqrt(kappa))^n
    mu == 0 && @warn "Momentum is tuned for strongly convex functions!"

    if verbose
        @info "🐱 Example file: worst-case performance of the accelerated gradient method" 
        @info "💻  PEPit guarantee: f(x_n)-f_*  <= $(round(PEPit_tau, digits=6)) (f(x_0) - f(x_*) + mu/2*||x_0 - x_*||^2)"
        @info "📝 Theoretical guarantee: f(x_n)-f_*  <= $(round(theoretical_tau, digits=6)) (f(x_0) - f(x_*) + mu/2*||x_0 - x_*||^2)"
    end

    return PEPit_tau, theoretical_tau

end

PEPit_val, theoretical_val = wc_accelerated_gradient_strongly_convex(0.1, 1.0, 2, verbose=true)

```

Key steps in the PEP workflow:

1. Create a `PEP()` instance.
2. Declare function/operator classes with `declare_function!`.
3. Create points and initial conditions.
4. Run algorithmic steps to define iterates.
5. Define a performance metric.
6. Call `solve!` to get the worst-case bound.

## Step-by-Step tutorial

Please take a look at the step-by-step tutorial for the minimal example that we just saw for accelerated gradient method for smooth strongly convex minimization below:

[Tutorial 1: Accelerated gradient method for smooth strongly convex minimization](https://github.com/PerformanceEstimation/PEPit.jl/blob/main/tutorials/unconstrained_convex_minimization/accelerated_gradient_strongly_convex/accelerated_gradient_strongly_convex.tutorial.ipynb)

You can find other tutorials in the `tutorials` folder of the package. 

## Examples

We also have many examples in the `examples` folder of the package, please take a look.

## Solvers

`PEPit.jl` uses `JuMP` and builds an SDP internally. The default backend in `solve!` is `Clarabel.Optimizer`, and you can pass another JuMP-compatible SDP solver through the `solver` keyword (for example, `Mosek.Optimizer` if you `using Mosek`).

Currently documented/used solvers in this repository are `Mosek` and `Clarabel`. Note: using `Mosek` requires a valid license (free for academic use), while `Clarabel` is open-source.

## Explicit dual certificates

`solve_dual!(problem; solver=Clarabel.Optimizer)` builds the same primal SDP, constructs its explicit conic dual with `Dualization.jl`, solves that dual model, and returns a `DualPEPCertificate`.

The certificate exposes PEP-level dual fields:

```julia
certificate = solve_dual!(problem; verbose=false)
certificate.α          # performance-metric multipliers
certificate.λ          # problem-level inequality multipliers
certificate.ν          # problem-level equality multipliers
certificate.θ          # function/class scalar-constraint multipliers
certificate.S          # main Gram PSD multiplier
certificate.Y.problem  # problem-level PSD block multipliers
certificate.Y.class    # function/class PSD block multipliers
```

After `solve_dual!`, `eval_dual(::Constraint)` and `eval_dual(::PSDMatrix)` are populated from the explicit-dual solution. Optimal dual solutions need not be unique, so the multipliers returned by `solve!` and `solve_dual!` should not be expected to match entrywise; their optimal objective values should agree up to solver tolerances.

## Repository layout

- `src/`: core implementation
  - `core/`: points, expressions, constraints, PSD matrices, and PEP assembly
  - `functions/`: function class definitions (smooth, convex, etc.)
  - `operators/`: operator class definitions (monotone, nonexpansive, etc.)
  - `primitive_steps/`: algorithmic building blocks (gradient, proximal, line search)
- `examples/`: runnable scripts for specific algorithms and settings
- `tutorials/`: literate tutorials (Markdown, Jupyter, LaTeX/PDF)
- `test/`: unit tests for core machinery

## Running examples

Examples are standard Julia scripts. For instance, you can run them as:

```julia
include("examples/unconstrained_convex_minimization/gradient_exact_line_search.jl")
```

## Notes and scope

This is a research-oriented tool aimed at building and solving PEPs. The API is still evolving, but the structure is already suitable for reproducing standard worst-case analyses for first-order methods and monotone inclusion algorithms.

## Contact

Please report any issues via the `Github Issue Tracker`. All types of issues are welcome including bug reports, feature requests, implementation for a specific research problem and so on. Also, please feel free to send an email :email: to [sd158@rice.edu](mailto:sd158@rice.edu) or [adrien.taylor@inria.fr](mailto:adrien.taylor@inria.fr) if you want to say hi :rocket:!	

## References 

[1] Y. Drori, M. Teboulle (2014). [Performance of first-order methods for smooth convex minimization: a novel approach](https://arxiv.org/pdf/1206.3209.pdf). Mathematical Programming 145(1–2), 451–482.

[2] A. Taylor, J. Hendrickx, F. Glineur (2017). [Smooth strongly convex interpolation and exact worst-case performance of first-order methods](https://arxiv.org/pdf/1502.05666.pdf). Mathematical Programming, 161(1-2), 307-345.

[3] A. Taylor, J. Hendrickx, F. Glineur (2017). [Exact worst-case performance of first-order methods for composite convex optimization](https://arxiv.org/pdf/1512.07516.pdf). SIAM Journal on Optimization, 27(3):1283–1313.

[4] B Goujaud, C. Moucer, F. Glineur, J.M. Hendrickx, A.B. Taylor, A. Dieuleveut (2024). [PEPit: computer-assisted worst-case analyses of first-order optimization methods in Python](https://arxiv.org/pdf/2201.04040). Mathematical Programming Computation 16 (3), 337-367.

[5] A. Taylor, J. Hendrickx, F. Glineur (2017). [Performance Estimation Toolbox (PESTO): automated worst-case analysis of first-order optimization methods](https://adrientaylor.github.io/share/PESTO_CDC_2017.pdf). In 56th IEEE Conference on Decision and Control (CDC).

[6] M. Lubin, O. Dowson,  J.D. Garcia, J. Huchette, B. Legat, J.P. Vielma . [JuMP 1.0: Recent improvements to a modeling language for mathematical optimization.](https://arxiv.org/pdf/2206.03866). Mathematical Programming Computation. 2023 Sep;15(3):581-9.
