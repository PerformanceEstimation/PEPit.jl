using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_flow_strongly_convex(mu; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_flow_strongly_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for a **gradient** flow.
That is, it computes the smallest possible $\tau(\mu)$ such that the guarantee

```math
\frac{d}{dt}\mathcal{V}(X_t) \leqslant -\tau(\mu)\mathcal{V}(X_t) ,
```

is valid, where $\mathcal{V}(X_t) = f(X_t) - f(x_\star)$, $X_t$ is the output of
the **gradient** flow, and where $x_\star$ is the minimizer of $f$.
In short, for given values of $\mu$, $\tau(\mu)$ is computed as the worst-case value
of the derivative $f(X_t)-f_\star$ when $f(X_t) -  f(x_\star)\leqslant 1$.

# Algorithm

For $t \geqslant 0$,

```math
\frac{d}{dt}X_t = -\nabla f(X_t),
```

with some initialization $X_{0}\triangleq x_0$.

# Theoretical guarantee


    The following **tight** guarantee can be found in [1, Proposition 11]:

```math
\frac{d}{dt}\mathcal{V}(X_t) \leqslant -2\mu\mathcal{V}(X_t).
```

    The detailed approach using PEPs is available in [2, Theorem 2.1].

# References


[[1] D. Scieur, V. Roulet, F. Bach and A. D'Aspremont (2017).
Integration methods and accelerated optimization algorithms.
In Advances in Neural Information Processing Systems (NIPS).](https://papers.nips.cc/paper/2017/file/bf62768ca46b6c3b5bea9515d1a1fc45-Paper.pdf)

[[2] C. Moucer, A. Taylor, F. Bach (2022).
A systematic approach to Lyapunov analyses of continuous-time models in convex optimization.](https://arxiv.org/pdf/2205.12772.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_flow_strongly_convex(0.1; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_gradient_flow_strongly_convex(mu; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu)
    func = declare_function!(problem, StronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)


    xt_dot = -gt


    lyap = ft - fs
    lyap_dot = gt * xt_dot


    set_initial_condition!(problem, lyap == 1)


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = -2 * mu
    mu == 0 && @warn "Momentum is tuned for strongly convex functions!"


    if verbose
        println("*** Example file: worst-case performance of the gradient flow ***")
        println("\tPEPit guarantee:\t d/dt[f(X_t)-f_*] <= $(round(pepit_tau, digits=6)) (f(X_t) - f(x_*))")
        println("\tTheoretical guarantee:\t d/dt[f(X_t)-f_*] <= $(round(theoretical_tau, digits=6)) (f(X_t) - f(x_*))")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_gradient_flow_strongly_convex(0.1; solver=Clarabel.Optimizer, verbose=true)
