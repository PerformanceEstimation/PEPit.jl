using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_inexact_gradient_descent(L, mu, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_inexact_gradient_descent`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for the **inexact gradient** method.
That is, it computes the smallest possible $\tau(n, L, \mu, \varepsilon)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu, \varepsilon) (f(x_0) - f_\star)
```

is valid, where $x_n$ is the output of the **inexact gradient** method,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$, $\mu$ and $\varepsilon$,
$\tau(n, L, \mu, \varepsilon)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $f(x_0) - f_\star \leqslant 1$.

# Algorithm


```math
x_{t+1} = x_t - \gamma d_t
```

    with

```math
\|d_t - \nabla f(x_t)\| \leqslant  \varepsilon \|\nabla f(x_t)\|
```

    and

```math
\gamma = \frac{2}{L_{\varepsilon} + \mu_{\varepsilon}}
```

    where $L_{\varepsilon} = (1 + \varepsilon) L$ and $\mu_{\varepsilon} = (1 - \varepsilon) \mu$.

# Theoretical guarantee


The **tight** worst-case guarantee obtained in [1, Theorem 5.3] or [2, Remark 1.6] is

```math
f(x_n) - f_\star \leqslant \left(\frac{L_{\varepsilon}-\mu_{\varepsilon}}{L_{\varepsilon}+\mu_{\varepsilon}}\right)^{2n}(f(x_0) - f_\star),
```

where tightness is achieved on simple quadratic functions.

# References
The detailed analyses can be found in [1, 2].

[[1] E. De Klerk, F. Glineur, A. Taylor (2020).
Worst-case convergence analysis of inexact gradient
and Newton methods through semidefinite programming performance estimation.
SIAM Journal on Optimization, 30(3), 2053-2082.](https://arxiv.org/pdf/1709.05191.pdf)

[[2] O. Gannot (2021).
A frequency-domain analysis of inexact gradient methods.
Mathematical Programming.](https://arxiv.org/pdf/1912.13494.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `epsilon`: level of inaccuracy.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_inexact_gradient_descent(1.0, 0.1, 0.1, 2; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_inexact_gradient_descent(L, mu, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, value!(func, x0) - fs <= 1)


    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    gamma = 2 / (Leps + meps)

    x = x0
    local dx, fx
    for i in 1:n
        x, dx, fx = inexact_gradient_step!(x, func, gamma, epsilon; notion="relative")
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = ((Leps - meps) / (Leps + meps))^(2 * n)

    if verbose
        println("*** Example file: worst-case performance of inexact gradient method in distance in function values ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_inexact_gradient_descent(1.0, 0.1, 0.1, 2; solver=Clarabel.Optimizer, verbose=true)
