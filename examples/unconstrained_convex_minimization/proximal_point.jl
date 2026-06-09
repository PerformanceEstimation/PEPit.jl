using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_proximal_point(gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_proximal_point`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is closed, proper, and convex (and potentially non-smooth).

# Performance metric

This code computes a worst-case guarantee for the **proximal point method** with step-size $\gamma$.
That is, it computes the smallest possible $\tau(n,\gamma)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, \gamma)  \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the proximal point method, and where $x_\star$ is a
minimizer of $f$.

In short, for given values of $n$ and $\gamma$,
$\tau(n,\gamma)$ is computed as the worst-case value of $f(x_n)-f_\star$
when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm


The proximal point method is described by

```math
x_{t+1} = \arg\min_x \left\{f(x)+\frac{1}{2\gamma}\|x-x_t\|^2 \right\},
```

where $\gamma$ is a step-size.

# Theoretical guarantee


The **tight** theoretical guarantee can be found in [1, Theorem 4.1]:

```math
f(x_n)-f_\star \leqslant \frac{\|x_0-x_\star\|^2}{4\gamma n},
```

where tightness is obtained on, e.g., one-dimensional linear problems on the positive orthant.

# References


[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex optimization.
SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

# Arguments
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_proximal_point(3, 4; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.020833, 0.020833)
```
"""
function wc_proximal_point(gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    fx = fs
    for _ in 1:n
        x, _, fx = proximal_step!(x, func, gamma)
    end


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 1 / (4 * gamma * n)


    if verbose
        println("*** Example file: worst-case performance of proximal point method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_proximal_point(3, 4; solver=Clarabel.Optimizer, verbose=true)
