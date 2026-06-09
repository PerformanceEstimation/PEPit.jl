using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_flow_convex(t; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_flow_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is convex.

# Performance metric

This code computes a worst-case guarantee for a **gradient** flow.
That is, it verifies the following inequality

```math
\frac{d}{dt}\mathcal{V}(X_t, t) \leqslant 0,
```

is valid, where $\mathcal{V}(X_t, t) = t(f(X_t) - f(x_\star)) + \frac{1}{2} \|X_t - x_\star\|^2$,
$X_t$ is the output of the **gradient** flow, and where $x_\star$ is the minimizer of $f$.
In short, for given values of $t$, it verifies $\frac{d}{dt}\mathcal{V}(X_t, t)\leqslant 0$.

# Algorithm

For $t \geqslant 0$,

```math
\frac{d}{dt}X_t = -\nabla f(X_t),
```

with some initialization $X_{0}\triangleq x_0$.

# Theoretical guarantee


    The following **tight** guarantee can be found in [1, p. 7]:

```math
\frac{d}{dt}\mathcal{V}(X_t, t) \leqslant 0.
```

    After integrating between $0$ and $T$,

```math
f(X_T) - f_\star \leqslant \frac{1}{2T}\|x_0 - x_\star\|^2.
```

    The detailed approach using PEPs is available in [2, Theorem 2.3].


# References


[[1] W. Su, S. Boyd, E. J. Candes (2016).
A differential equation for modeling Nesterov's accelerated gradient method: Theory and insights.
In the Journal of Machine Learning Research (JMLR).](https://jmlr.org/papers/volume17/15-084/15-084.pdf)

[[2] C. Moucer, A. Taylor, F. Bach (2022).
A systematic approach to Lyapunov analyses of continuous-time models in convex optimization.](https://arxiv.org/pdf/2205.12772.pdf)

# Arguments
- `t`: time step
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_flow_convex(2.5; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.0, 0.0)
```
"""
function wc_gradient_flow_convex(t; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)


    xt_dot = -gt


    lyap_dot = (ft - fs) + t * gt * xt_dot + (xt - xs) * xt_dot


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 0.0


    if verbose
        println("*** Example file: worst-case performance of the gradient flow ***")
        println("\tPEPit guarantee:\t d/dt V(X_t) <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t d/dt V(X_t) <= $(round(theoretical_tau, digits=6))")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_gradient_flow_convex(2.5; solver=Clarabel.Optimizer, verbose=true)
