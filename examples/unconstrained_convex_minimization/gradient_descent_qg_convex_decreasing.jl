using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_qg_convex_decreasing(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_qg_convex_decreasing`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is quadratically upper bounded ($\text{QG}^+$ [1]), i.e.
$\forall x, f(x) - f_\star \leqslant \frac{L}{2} \|x-x_\star\|^2$, and convex.

# Performance metric

This code computes a worst-case guarantee for **gradient descent** with decreasing step-sizes.
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L) \| x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of gradient descent with decreasing step-sizes, and
where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$ and $L$,
$\tau(n, L)$ is computed as the worst-case
value of $f(x_n)-f_\star$ when $||x_0 - x_\star||^2 \leqslant 1$.

# Algorithm

Gradient descent with decreasing step sizes is described by

```math
x_{t+1} = x_t - \gamma_t \nabla f(x_t)
```

with

```math
\gamma_t = \frac{1}{L u_{t+1}}
```

where the sequence $u$ is defined by

```math
\begin{aligned}
    u_0 & = & 1 \\
    u_{t} & = & \frac{u_{t-1}}{2} + \sqrt{\left(\frac{u_{t-1}}{2}\right)^2 + 2}, \quad \mathrm{for } t \geq 1
\end{aligned}
```
# Theoretical guarantee

The **tight** theoretical guarantee is conjectured in [1, Conjecture A.3]:

```math
f(x_n)-f_\star \leqslant \frac{L}{2 u_t} \|x_0-x_\star\|^2.
```

# References
No bibliographic reference was listed in the corresponding Python PEPit example docstring.

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
pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex_decreasing(1.0, 6; verbose=true)
```
"""
function wc_gradient_descent_qg_convex_decreasing(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, ConvexQGFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x = set_initial_point!(problem)
    g, f = oracle!(func, x)


    set_initial_condition!(problem, (x - xs)^2 <= 1)


    u = 1.0
    for i in 1:n

        u = u / 2 + sqrt((u / 2)^2 + 2)
        gamma = 1 / (L * u)
        x = x - gamma * g
        g, f = oracle!(func, x)
    end


    theoretical_tau = L / (2 * u)


    set_performance_metric!(problem, f - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical conjecture:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex_decreasing(1.0, 6; verbose=true)
