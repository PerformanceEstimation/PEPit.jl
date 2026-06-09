using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_silver_stepsize_convex(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_silver_stepsize_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code computes a worst-case guarantee for $n$ steps of the **gradient descent** method tuned
according to the silver stepsize schedule.
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of gradient descent using the silver step-sizes, and
where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$, and $L$, $\tau(n, L)$ is computed as the worst-case
value of $f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

Gradient descent is described by

```math
x_{t+1} = x_t - \gamma_t \nabla f(x_t),
```

where $\gamma_t$ is a step-size of the $t^{th}$ step of the silver step-size schedule described in [1].

# Theoretical guarantee

The theoretical guarantee for the convergence rate of the silver stepsize can be found in [1, Theorem 1.1]:

```math
f(x_n)-f_\star \leqslant \frac{L}{1 + \sqrt{4(1 + \sqrt{2})^{2k}-3}} \|x_0-x_\star\|^2,
```

where $k$ is such that $n = 2^k - 1$.

# References


[[1] J. M. Altschuler, P. A. Parrilo (2023).
Acceleration by Stepsize Hedging II: Silver Stepsize Schedule for Smooth Convex Optimization.
arXiv preprint arXiv:2309.16530.](https://arxiv.org/abs/2309.16530)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_convex(10.0, 7; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.184215, 0.343775)
```
"""
function wc_gradient_descent_silver_stepsize_convex(L, n; solver=Clarabel.Optimizer, verbose=true)

    k = log2(n + 1)
    if !isinteger(k)
        @warn "Silver step-size strategy is only defined when n is a power of 2 minus 1." *
              " The provided input n is not a power of 2 minus 1." *
              " n is reset as the largest acceptable value smaller than the provided one."
        k = floor(Int, k)
        n = 2^k - 1
    end


    fast_dyadic_valuation(i) = trailing_zeros(i)
    h = [1 + (1 + sqrt(2))^(fast_dyadic_valuation(i) - 1) for i in 1:n]


    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 1:n
        x = x - h[i] / L * gradient!(func, x)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (1 + sqrt(4 * (1 + sqrt(2))^(2 * k) - 3))


    if verbose
        println("*** Example file: worst-case performance of gradient descent with silver step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_convex(10.0, 7; solver=Clarabel.Optimizer, verbose=true)
