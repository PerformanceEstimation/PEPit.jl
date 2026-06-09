using PEPit
using OrderedCollections

@doc raw"""
    wc_frank_wolfe(L, D, R, center, n; verbose::Bool=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_frank_wolfe`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f_1(x) + f_2(x)\},
```

where $f_1$ is $L$-smooth and convex
and where $f_2$ is a convex indicator function on $\mathcal{D}$
of diameter at most $D$ and radius at most $R$ around `center`.

# Performance metric

This code computes a worst-case guarantee for the **conditional gradient** method, aka **Frank-Wolfe** method.
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
F(x_n) - F(x_\star) \leqslant \tau(n, L, D, R),
```

is valid, where $x_n$ is the output of the **conditional gradient** method,
and where $x_\star$ is a minimizer of $F$.
In short, for given values of $n$ and $L$, $\tau(n, L, D, R)$
is computed as the worst-case value of $F(x_n) - F(x_\star)$.

# Algorithm


This method was first presented in [1]. A more recent version can be found in, e.g., [2, Algorithm 1].
For $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        y_t & = & \arg\min_{s \in \mathcal{D}} \langle s \mid \nabla f_1(x_t) \rangle, \\
        x_{t+1} & = & \frac{t}{t + 2} x_t + \frac{2}{t + 2} y_t.
    \end{aligned}
```
# Theoretical guarantee


An **upper** guarantee obtained in [2, Theorem 1] when R = infinity is

```math
F(x_n) - F(x_\star) \leqslant \frac{2L D^2}{n+2}.
```

# References


[[1] M .Frank, P. Wolfe (1956).
An algorithm for quadratic programming.
Naval research logistics quarterly, 3(1-2), 95-110.](https://arxiv.org/pdf/1608.04826.pdf)

[[2] M. Jaggi (2013).
Revisiting Frank-Wolfe: Projection-free sparse convex optimization.
In 30th International Conference on Machine Learning (ICML).](http://proceedings.mlr.press/v28/jaggi13.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `D`: diameter of $\mathcal{D}$.
- `R`: radius of $\mathcal{D}$.
- `center`: center of $\mathcal{D}$. If None, the radius constraint must be observed to one center.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
PEPit_tau, theoretical_tau = wc_frank_wolfe(1.0, 1.0, Inf, nothing, 10; verbose=true)
# Returns approximately: (PEPit_tau, theoretical_tau) = (0.07829, 0.166667)
```
"""
function wc_frank_wolfe(L, D, R, center, n; verbose::Bool=true)

    problem = PEP()


    f1 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)
    f2 = declare_function!(problem, ConvexIndicatorFunction,
        OrderedDict("D" => D, "R" => R, "center" => center); reuse_gradient=false)


    F = f1 + f2


    xs = stationary_point!(F)
    fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    _ = value!(f1, x0)
    _ = value!(f2, x0)


    x = x0
    for t in 0:(n-1)
        g = gradient!(f1, x)
        y, _, _ = linear_optimization_step!(g, f2)
        λ = 2 / (t + 2)
        x = (1 - λ) * x + λ * y
    end


    set_performance_metric!(problem, value!(F, x) - fs)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = 2 * L * D^2 / (n + 2)

    if verbose
        println("*** Example file: worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(PEPit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end

    return PEPit_tau, theoretical_tau


end

PEPit_tau, theoretical_tau = wc_frank_wolfe(1.0, 1.0, Inf, nothing, 10; verbose=true)
