using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_qg_convex(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_qg_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is quadratically upper bounded ($\text{QG}^+$ [1]), i.e.
$\forall x, f(x) - f_\star \leqslant \frac{L}{2} \|x-x_\star\|^2$, and convex.

# Performance metric

This code computes a worst-case guarantee for **gradient descent** with fixed step-size $\gamma$.
That is, it computes the smallest possible $\tau(n, L, \gamma)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \gamma) \| x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of gradient descent with fixed step-size $\gamma$, and
where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$, $L$,
and $\gamma$, $\tau(n, L, \gamma)$ is computed as the worst-case
value of $f(x_n)-f_\star$ when $||x_0 - x_\star||^2 \leqslant 1$.

# Algorithm

Gradient descent is described by

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size.

# Theoretical guarantee

When $\gamma < \frac{1}{L}$, the **lower** theoretical guarantee can be found in [1, Theorem 2.2]:

```math
f(x_n)-f_\star \leqslant \frac{L}{2}\max\left(\frac{1}{2n L \gamma + 1}, L \gamma\right) \|x_0-x_\star\|^2.
```

# References


The detailed approach is available in [1, Theorem 2.2].

[[1] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
Optimal first-order methods for convex functions with a quadratic upper bound.](https://arxiv.org/pdf/2205.15033.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex(L, 0.2 / L, 4; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.192308, 0.192308)
```
"""
function wc_gradient_descent_qg_convex(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, ConvexQGFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 1:n
        x = x - gamma * gradient!(func, x)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / 2 * max(1 / (2 * n * L * gamma + 1), L * gamma)


    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex(L, 0.2 / L, 4; verbose=true)
