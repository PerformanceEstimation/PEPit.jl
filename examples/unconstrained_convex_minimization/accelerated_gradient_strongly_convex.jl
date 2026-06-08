using PEPit
using OrderedCollections
using JuMP

@doc raw"""
    wc_accelerated_gradient_strongly_convex(mu, L, n; verbose=false)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_gradient_strongly_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for an **accelerated gradient** method, a.k.a **fast gradient** method.
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu) \left(f(x_0) -  f(x_\star) + \frac{\mu}{2}\|x_0 - x_\star\|^2\right),
```

is valid, where $x_n$ is the output of the **accelerated gradient** method,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $f(x_0) -  f(x_\star) + \frac{\mu}{2}\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

For $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        y_t & = & x_t + \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}(x_t - x_{t-1}) \\
        x_{t+1} & = & y_t - \frac{1}{L} \nabla f(y_t)
    \end{aligned}
```
with $x_{-1}:= x_0$.

# Theoretical guarantee


    The following **upper** guarantee can be found in [1,  Corollary 4.15]:

```math
f(x_n)-f_\star \leqslant \left(1 - \sqrt{\frac{\mu}{L}}\right)^n \left(f(x_0) -  f(x_\star) + \frac{\mu}{2}\|x_0 - x_\star\|^2\right).
```

# References


[[1] A. d'Aspremont, D. Scieur, A. Taylor (2021). Acceleration Methods. Foundations and Trends
in Optimization: Vol. 5, No. 1-2.](https://arxiv.org/pdf/2101.09545.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
PEPit_val, theoretical_val = wc_accelerated_gradient_strongly_convex(0.1, 1.0, 2, verbose=true)
```
"""
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

    @info "🐯 Detailed results"

    res = solve!(problem; verbose=false, return_full_model=true)

    @show res.wc_value
    @show res.model
    @show value.(res.variables.F)
    @show value.(res.variables.G)
    @show dual.(res.constraints.initial)
    @show dual.(res.constraints.class)
    @show dual.(res.constraints.performance)

    return PEPit_tau, theoretical_tau

end

PEPit_val, theoretical_val = wc_accelerated_gradient_strongly_convex(0.1, 1.0, 2, verbose=true)
