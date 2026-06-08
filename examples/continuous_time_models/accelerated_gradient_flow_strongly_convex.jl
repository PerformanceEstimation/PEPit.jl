using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_accelerated_gradient_flow_strongly_convex(mu; psd=true, solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_gradient_flow_strongly_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_{x\in\mathbb{R}^d} f(x),
```

where $f$ is $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for an **accelerated gradient** flow.
That is, it computes the smallest possible $\tau(\mu)$ such that the guarantee

```math
\frac{d}{dt}\mathcal{V}_{P}(X_t) \leqslant -\tau(\mu)\mathcal{V}_P(X_t) ,
```

is valid with

```math
\mathcal{V}_{P}(X_t) = f(X_t) - f(x_\star) + (X_t - x_\star, \frac{d}{dt}X_t)^T(P \otimes I_d)(X_t - x_\star, \frac{d}{dt}X_t) ,
```

where $I_d$ is the identity matrix, $X_t$ is the output of an **accelerated gradient** flow,
and where $x_\star$ is the minimizer of $f$.

In short, for given values of $\mu$, $\tau(\mu)$ is computed as the worst-case value of
the derivative of $f(X_t)-f_\star$ when $f(X_t) -  f(x_\star)\leqslant 1$.

# Algorithm

For $t \geqslant 0$,

```math
\frac{d^2}{dt^2}X_t + 2\sqrt{\mu}\frac{d}{dt}X_t + \nabla f(X_t) = 0,
```

with some initialization $X_{0}\triangleq x_0$.

# Theoretical guarantee


    The following **tight** guarantee for $P = \frac{1}{2}\begin{pmatrix} \mu & \sqrt{\mu} \\ \sqrt{\mu} & 1\end{pmatrix}$,
    for which $\mathcal{V}_{P} \geqslant 0$ can be found in [1, Appendix B], [2, Theorem 4.3]:

```math
\frac{d}{dt}\mathcal{V}_P(X_t) \leqslant -\sqrt{\mu}\mathcal{V}_P(X_t).
```

    For $P = \begin{pmatrix} \frac{4}{9}\mu & \frac{4}{3}\sqrt{\mu} \\ \frac{4}{3}\sqrt{\mu} & \frac{1}{2}\end{pmatrix}$,
    for which $\mathcal{V}_{P}(X_t) \geqslant 0$ along the trajectory, the following **tight** guarantee can
    be found in [3, Corollary 2.5],

```math
\frac{d}{dt}\mathcal{V}_P(X_t) \leqslant -\frac{4}{3}\sqrt{\mu}\mathcal{V}_P(X_t).
```


# References


[[1] A. C. Wilson, B. Recht, M. I. Jordan (2021).
A Lyapunov analysis of accelerated methods in optimization.
In the Journal of Machine Learning Reasearch (JMLR), 22(113):1-34, 2021.](https://jmlr.org/papers/volume22/20-195/20-195.pdf)

[[2] J.M. Sanz-Serna and K. C. Zygalakis (2021).
The connections between Lyapunov functions for some optimization algorithms and differential equations.
In SIAM Journal on Numerical Analysis, 59 pp 1542-1565.](https://arxiv.org/pdf/2009.00673.pdf)

[[3] C. Moucer, A. Taylor, F. Bach (2022).
A systematic approach to Lyapunov analyses of continuous-time models in convex optimization.
In SIAM Journal on Optimization 33 (3), 1558-1586.](https://arxiv.org/pdf/2205.12772.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `psd`: option for positivity of $P$ in the Lyapunov function $\mathcal{V}_{P}$
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_strongly_convex(0.1; psd=true, solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_accelerated_gradient_flow_strongly_convex(mu; psd=true, solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu)
    func = declare_function!(problem, StronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)
    xt_dot = set_initial_point!(problem)


    xt_dot_dot = -2 * sqrt(mu) * xt_dot - gt


    if psd
        lyap = ft - fs + 1 / 2 * (sqrt(mu) * (xt - xs) + xt_dot)^2
        lyap_dot = gt * xt_dot + (sqrt(mu) * (xt - xs) + xt_dot) * (sqrt(mu) * xt_dot + xt_dot_dot)
    else
        lyap = ft - fs + mu * 4 / 9 * (xt - xs)^2 + 2 * 2 / 3 * sqrt(mu) * (xt - xs) * xt_dot + 1 / 2 * xt_dot^2
        lyap_dot = gt * xt_dot + mu * 8 / 9 * (xt - xs) * xt_dot + 4 / 3 * sqrt(mu) * (xt_dot^2 + (xt - xs) * xt_dot_dot) + xt_dot_dot * xt_dot
    end


    set_initial_condition!(problem, lyap == 1)


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if psd
        theoretical_tau = -sqrt(mu)
    else
        theoretical_tau = -4 / 3 * sqrt(mu)
    end
    mu == 0 && @warn "Momentum is tuned for strongly convex functions!"


    if verbose
        println("*** Example file: worst-case performance of an accelerated gradient flow ***")
        println("\tPEPit guarantee:\t d/dt V(X_t,t) <= $(round(pepit_tau, digits=6)) V(X_t,t)")
        println("\tTheoretical guarantee:\t d/dt V(X_t) <= $(round(theoretical_tau, digits=6)) V(X_t,t)")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_strongly_convex(0.1; psd=true, solver=Clarabel.Optimizer, verbose=true)
