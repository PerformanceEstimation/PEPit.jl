using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_accelerated_gradient_flow_convex(t; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_gradient_flow_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is convex.

# Performance metric

This code computes a worst-case guarantee for an **accelerated gradient** flow.
That is, it verifies the inequality

```math
\frac{d}{dt}\mathcal{V}(X_t, t) \leqslant 0 ,
```

is valid, where $\mathcal{V}(X_t, t) = t^2(f(X_t) - f(x_\star)) + 2 \|(X_t - x_\star) + \frac{t}{2}\frac{d}{dt}X_t \|^2$,
$X_t$ is the output of an **accelerated gradient** flow, and where $x_\star$ is the minimizer of $f$.

In short, for given values of $t$, it verifies $\frac{d}{dt}\mathcal{V}(X_t, t) \leqslant 0$.

# Algorithm

For $t \geqslant 0$,

```math
\frac{d^2}{dt^2}X_t + \frac{3}{t}\frac{d}{dt}X_t + \nabla f(X_t) = 0,
```

with some initialization $X_{0}\triangleq x_0$.

# Theoretical guarantee


    The following **tight** guarantee can be verified in [1, Section 2]:

```math
\frac{d}{dt}\mathcal{V}(X_t, t) \leqslant 0.
```

    After integrating between $0$ and $T$,

```math
f(X_T) - f_\star \leqslant \frac{2}{T^2}\|x_0 - x_\star\|^2.
```

    The detailed approach using PEPs is available in [2, Theorem 2.6].

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
pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_convex(3.4; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (-0.0, 0.0)
```
"""
function wc_accelerated_gradient_flow_convex(t; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)
    xt_dot = set_initial_point!(problem)


    xt_dot_dot = -3 / t * xt_dot - gt


    lyap = t^2 * (ft - fs) + 2 * ((xt - xs) + t / 2 * xt_dot)^2
    lyap_dot = 2 * t * (ft - fs) + t^2 * xt_dot * gt + 4 * ((xt - xs) + t / 2 * xt_dot) * (3 / 2 * xt_dot + t / 2 * xt_dot_dot)


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 0.0


    if verbose
        println("*** Example file: worst-case performance of an accelerated gradient flow ***")
        println("\tPEPit guarantee:\t d/dt V(X_t,t) <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t d/dt V(X_t) <= $(round(theoretical_tau, digits=6))")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_convex(3.4; solver=Clarabel.Optimizer, verbose=true)
