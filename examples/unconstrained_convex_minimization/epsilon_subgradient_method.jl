using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_epsilon_subgradient_method(M, n, gamma, eps, R; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_epsilon_subgradient_method`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is closed, convex, and proper. This problem is a (possibly non-smooth) minimization problem.

# Performance metric

This code computes a worst-case guarantee for the $\varepsilon$ **-subgradient method**. That is, it computes
the smallest possible $\tau(n, M, \gamma, \varepsilon, R)$ such that the guarantee

```math
\min_{0 \leqslant t \leqslant n} f(x_t) - f_\star \leqslant \tau(n, M, \gamma, \varepsilon, R)
```

is valid, where $x_t$ are the iterates of the $\varepsilon$ **-subgradient method**
after $t\leqslant n$ steps,
where $x_\star$ is a minimizer of $f$, where $M$ is an upper bound on the norm of all
$\varepsilon$-subgradients encountered, and when $\|x_0-x_\star\|\leqslant R$.

In short, for given values of $M$, of the accuracy $\varepsilon$, of the step-size $\gamma$,
of the initial distance $R$, and of the number of iterations $n$,
$\tau(n, M, \gamma, \varepsilon, R)$ is computed as the worst-case value of
$\min_{0 \leqslant t \leqslant n} f(x_t) - f_\star$.

# Algorithm

For $t\in \{0, \dots, n-1 \}$

```math
    \begin{aligned}
        g_{t} & \in & \partial_{\varepsilon} f(x_t) \\
        x_{t+1} & = & x_t - \gamma g_t
    \end{aligned}
```
# Theoretical guarantee
An upper bound is obtained in [1, Lemma 2]:

```math
\min_{0 \leqslant t \leqslant n} f(x_t)- f(x_\star) \leqslant \frac{R^2+2(n+1)\gamma\varepsilon+(n+1) \gamma^2 M^2}{2(n+1) \gamma}.
```

# References


[[1] R.D. Millan, M.P. Machado (2019).
Inexact proximal epsilon-subgradient methods for composite convex optimization problems.
Journal of Global Optimization 75.4 (2019): 1029-1060.](https://arxiv.org/pdf/1805.10120.pdf)

# Arguments
- `M`: the bound on norms of epsilon-subgradients.
- `n`: number of iterations.
- `gamma`: step-size parameter.
- `eps`: the bound on the value of epsilon (inaccuracy).
- `R`: the bound on initial distance to an optimal solution.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_epsilon_subgradient_method(M, n, gamma, eps, R; verbose=true)
```
"""
function wc_epsilon_subgradient_method(M, n, gamma, eps, R; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= R^2)


    x = x0
    for _ in 1:n
        x, gx, fx, epsilon = epsilon_subgradient_step!(x, func, gamma)
        set_performance_metric!(problem, fx - fs)
        add_constraint!(problem, epsilon <= eps)
        add_constraint!(problem, gx^2 <= M^2)
    end


    gx, fx = oracle!(func, x)
    add_constraint!(problem, gx^2 <= M^2)
    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (R^2 + 2 * (n + 1) * gamma * eps + (n + 1) * gamma^2 * M^2) / (2 * (n + 1) * gamma)

    if verbose
        println("*** Example file: worst-case performance of the epsilon-subgradient method ***")
        println("\tPEPit guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(theoretical_tau, digits=6))")
    end


    return pepit_tau, theoretical_tau
end


M, n, eps, R = 2, 6, 0.1, 1
gamma = 1 / sqrt(n + 1)
pepit_tau, theoretical_tau = wc_epsilon_subgradient_method(M, n, gamma, eps, R; verbose=true)
