using PEPit
using OrderedCollections

@doc raw"""
    wc_inexact_gradient_exact_line_search(L, mu, epsilon, n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_inexact_gradient_exact_line_search`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for an **inexact gradient method with exact linesearch (ELS)**.
That is, it computes the smallest possible $\tau(n, L, \mu, \varepsilon)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu, \varepsilon) ( f(x_0) - f_\star )
```

is valid, where $x_n$ is the output of the **gradient descent with an inexact descent direction
and an exact linesearch**, and where $x_\star$ is the minimizer of $f$.

The inexact descent direction $d$ is assumed to satisfy a relative inaccuracy described by
(with $0 \leqslant \varepsilon < 1$)

```math
\|\nabla f(x_t) - d_t\| \leqslant \varepsilon \|\nabla f(x_t)\|,
```

where $\nabla f(x_t)$ is the true gradient, and $d_t$
is the approximate descent direction that is used.

# Algorithm


For $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        \gamma_t & = & \arg\min_{\gamma \in R^d} f(x_t- \gamma d_t), \\
        x_{t+1} & = & x_t - \gamma_t d_t.
    \end{aligned}
```
# Theoretical guarantee

The **tight** guarantee obtained in [1, Theorem 5.1] is

```math
f(x_n) - f_\star\leqslant \left(\frac{L_{\varepsilon} - \mu_{\varepsilon}}{L_{\varepsilon} + \mu_{\varepsilon}}\right)^{2n}( f(x_0) - f_\star ),
```

with $L_{\varepsilon} = (1 + \varepsilon) L$ and $\mu_{\varepsilon} = (1 - \varepsilon) \mu$.
Tightness is achieved on simple quadratic functions.

# References
The detailed approach (based on convex relaxations) is available in [1],

[[1]  E. De Klerk, F. Glineur, A. Taylor (2017). On the worst-case complexity of the gradient method with exact
line search for smooth strongly convex functions. Optimization Letters, 11(7), 1185-1199.](https://link.springer.com/content/pdf/10.1007/s11590-016-1087-4.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `epsilon`: level of inaccuracy.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_inexact_gradient_exact_line_search(1.0, 0.1, 0.1, 2; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.518917, 0.518917)
```
"""
function wc_inexact_gradient_exact_line_search(L, mu, epsilon, n; verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param; reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x = set_initial_point!(problem)
    set_initial_condition!(problem, value!(func, x) - fs <= 1)


    fx = value!(func, x)


    for i in 1:n
        _, dx, _ = inexact_gradient_step!(x, func, 0.0, epsilon; notion="relative")
        x, gx, fx = exact_linesearch_step!(x, func, [dx])
    end


    set_performance_metric!(problem, fx - fs)

    pepit_tau = solve!(problem; verbose=verbose)

    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    theoretical_tau = ((Leps - meps) / (Leps + meps))^(2 * n)

    if verbose
        println("*** Example file: worst-case performance of inexact gradient descent with exact linesearch ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_inexact_gradient_exact_line_search(1.0, 0.1, 0.1, 2; verbose=true)
