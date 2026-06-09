using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_inexact_accelerated_gradient(L, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_inexact_accelerated_gradient`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code computes a worst-case guarantee for an **accelerated gradient method** using **inexact first-order
information**. That is, it computes the smallest possible $\tau(n, L, \varepsilon)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \varepsilon)  \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of **inexact accelerated gradient descent** and where $x_\star$
is a minimizer of $f$.

The inexact descent direction is assumed to satisfy a relative inaccuracy described by
(with $0\leqslant \varepsilon \leqslant 1$)

```math
\|\nabla f(y_t) - d_t\| \leqslant \varepsilon \|\nabla f(y_t)\|,
```

where $\nabla f(y_t)$ is the true gradient at $y_t$ and $d_t$ is
the approximate descent direction that is used.

# Algorithm

The inexact accelerated gradient method of this example is provided by

```math
    \begin{aligned}
        x_{t+1} & = & y_t - \frac{1}{L} d_t\\
        y_{k+1} & = & x_{t+1} + \frac{t-1}{t+2} (x_{t+1} - x_t).
    \end{aligned}
```
# Theoretical guarantee

When $\varepsilon=0$, a **tight** empirical guarantee can be found in [1, Table 1]:

```math
f(x_n)-f_\star \leqslant \frac{2L\|x_0-x_\star\|^2}{n^2 + 5 n + 6},
```

which is achieved on some Huber loss functions (when $\varepsilon=0$).

# References


[[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Exact worst-case performance of first-order methods for composite
convex optimization. SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `epsilon`: level of inaccuracy
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_inexact_accelerated_gradient(1.0, 0.1, 5; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.039388, 0.035714)
```
"""
function wc_inexact_accelerated_gradient(L, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    y = x0
    for i in 0:(n - 1)
        x_old = x_new
        x_new, dy, fy = inexact_gradient_step!(y, func, 1 / L, epsilon; notion="relative")
        y = x_new + i / (i + 3) * (x_new - x_old)
    end
    _, fx = oracle!(func, x_new)


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 * L / (n^2 + 5 * n + 6)

    if verbose
        println("*** Example file: worst-case performance of inexact accelerated gradient method ***")
        println("\tPEPit guarantee:\t\t\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee for epsilon = 0 :\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_inexact_accelerated_gradient(1.0, 0.1, 5; solver=Clarabel.Optimizer, verbose=true)
