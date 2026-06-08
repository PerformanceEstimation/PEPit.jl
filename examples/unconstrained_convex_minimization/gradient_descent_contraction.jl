using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_contraction(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_contraction`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for **gradient descent** with fixed step-size $\gamma$.
That is, it computes the smallest possible $\tau(n, L, \mu, \gamma)$ such that the guarantee

```math
\| x_n - y_n \|^2 \leqslant \tau(n, L, \mu, \gamma) \| x_0 - y_0 \|^2
```

is valid, where $x_n$ and $y_n$ are the outputs of
the gradient descent method with fixed step-size $\gamma$,
starting respectively from $x_0$ and $y_0$.

In short, for given values of $n$, $L$, $\mu$ and $\gamma$,
$\tau(n, L, \mu \gamma)$ is computed as the worst-case value of $\| x_n - y_n \|^2$
when $\| x_0 - y_0 \|^2 \leqslant 1$.

# Algorithm

For $t\in\{0,1,\ldots,n-1\}$, gradient descent is described by

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size.

# Theoretical guarantee

The **tight** theoretical guarantee is

```math
\| x_n - y_n \|^2 \leqslant  \max\{(1-L\gamma)^2,(1-\mu \gamma)^2\}^n\| x_0 - y_0 \|^2,
```

which is tight on simple quadratic functions.

# References
No bibliographic reference was listed in the corresponding Python PEPit example docstring.

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent_contraction(L, mu, gamma, n; verbose=true)
```
"""
function wc_gradient_descent_contraction(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


    x0 = set_initial_point!(problem)
    y0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - y0)^2 <= 1)


    x = x0
    y = y0
    for _ in 1:n
        x = x - gamma * gradient!(func, x)
        y = y - gamma * gradient!(func, y)
    end


    set_performance_metric!(problem, (x - y)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max((1 - gamma * L)^2, (1 - gamma * mu)^2)^n

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes in contraction ***")
        println("\tPEPit guarantee:\t ||x_n - y_n||^2 <= $(round(pepit_tau, digits=6)) ||x_0 - y_0||^2")
        println("\tTheoretical guarantee:\t ||x_n - y_n||^2 <= $(round(theoretical_tau, digits=6)) ||x_0 - y_0||^2")
    end

    return pepit_tau, theoretical_tau
end


L = 1.0
mu = 0.1
gamma = 1 / L
n = 1
pepit_tau, theoretical_tau = wc_gradient_descent_contraction(L, mu, gamma, n; verbose=true)
