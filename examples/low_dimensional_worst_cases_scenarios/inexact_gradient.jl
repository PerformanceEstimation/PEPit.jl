using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_inexact_gradient(L, mu, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_inexact_gradient`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for an **inexact gradient method** and looks for a low-dimensional
worst-case example nearly achieving this worst-case guarantee using $10$ iterations of the logdet heuristic.

That is, it computes the smallest possible $\tau(n,L,\mu,\varepsilon)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n,L,\mu,\varepsilon) (f(x_0) - f_\star)
```

is valid, where $x_n$ is the output of the gradient descent with an inexact descent direction,
and where $x_\star$ is the minimizer of $f$. Then, it looks for a low-dimensional nearly achieving this
performance.

The inexact descent direction is assumed to satisfy a relative inaccuracy
described by (with $0 \leqslant \varepsilon \leqslant 1$)

```math
\|\nabla f(x_t) - d_t\| \leqslant \varepsilon \|\nabla f(x_t)\|,
```

where $\nabla f(x_t)$ is the true gradient,
and $d_t$ is the approximate descent direction that is used.

# Algorithm


The inexact gradient descent under consideration can be written as

```math
x_{t+1} = x_t - \frac{2}{L_{\varepsilon} + \mu_{\varepsilon}} d_t
```

where $d_t$ is the inexact search direction, $L_{\varepsilon} = (1 + \varepsilon)L$
and $\mu_{\varepsilon} = (1-\varepsilon) \mu$.

# Theoretical guarantee


A **tight** worst-case guarantee obtained in [1, Theorem 5.3] or [2, Remark 1.6] is

```math
f(x_n) - f_\star \leqslant \left(\frac{L_{\varepsilon} - \mu_{\varepsilon}}{L_{\varepsilon} + \mu_{\varepsilon}}\right)^{2n}(f(x_0) - f_\star ),
```

with $L_{\varepsilon} = (1 + \varepsilon)L$ and $\mu_{\varepsilon} = (1-\varepsilon) \mu$. This
guarantee is achieved on one-dimensional quadratic functions.

# References
The detailed analyses can be found in [1, 2]. The logdet heuristic is presented in [3].

[[1] E. De Klerk, F. Glineur, A. Taylor (2020). Worst-case convergence analysis of
inexact gradient and Newton methods through semidefinite programming performance estimation.
SIAM Journal on Optimization, 30(3), 2053-2082.](https://arxiv.org/pdf/1709.05191.pdf)

[[2] O. Gannot (2021). A frequency-domain analysis of inexact gradient methods.
Mathematical Programming.](https://arxiv.org/pdf/1912.13494.pdf)

[[3] F. Maryam, H. Hindi, S. Boyd (2003). Log-det heuristic for matrix rank minimization with applications to Hankel
and Euclidean distance matrices. American Control Conference (ACC).](https://web.stanford.edu/~boyd/papers/pdf/rank_min_heur_hankel.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `epsilon`: level of inaccuracy
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_inexact_gradient(1.0, 0.1, 0.1, 6; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_inexact_gradient(L, mu, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


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


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=10)


    theoretical_tau = ((Leps - meps) / (Leps + meps))^(2 * n)


    if verbose
        println("*** Example file: worst-case performance of inexact gradient ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* == $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_inexact_gradient(1.0, 0.1, 0.1, 6; solver=Clarabel.Optimizer, verbose=true)
