using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_optimized_gradient(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_optimized_gradient`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code computes a worst-case guarantee for **optimized gradient method** (OGM). That is, it computes
the smallest possible $\tau(n, L)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of OGM and where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$ and $L$, $\tau(n, L)$ is computed as the worst-case value
of $f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

The optimized gradient method is described by

```math
    \begin{aligned}
        x_{t+1} & = & y_t - \frac{1}{L} \nabla f(y_t)\\
        y_{t+1} & = & x_{t+1} + \frac{\theta_{t}-1}{\theta_{t+1}}(x_{t+1}-x_t)+\frac{\theta_{t}}{\theta_{t+1}}(x_{t+1}-y_t),
    \end{aligned}
```
with

```math
    \begin{aligned}
        \theta_0 & = & 1 \\
        \theta_t & = & \frac{1 + \sqrt{4 \theta_{t-1}^2 + 1}}{2}, \forall t \in [|1, n-1|] \\
        \theta_n & = & \frac{1 + \sqrt{8 \theta_{n-1}^2 + 1}}{2}.
    \end{aligned}
```
# Theoretical guarantee

The **tight** theoretical guarantee can be found in [2, Theorem 2]:

```math
f(x_n)-f_\star \leqslant \frac{L\|x_0-x_\star\|^2}{2\theta_n^2},
```

where tightness follows from [3, Theorem 3].

# References

The optimized gradient method was developed in [1, 2]; the corresponding lower bound was first obtained in [3].

[[1] Y. Drori, M. Teboulle (2014).
Performance of first-order methods for smooth convex minimization: a novel approach.
Mathematical Programming 145(1-2), 451-482.](https://arxiv.org/pdf/1206.3209.pdf)

[[2] D. Kim, J. Fessler (2016).
Optimized first-order methods for smooth convex minimization.
Mathematical Programming 159.1-2: 81-107.](https://arxiv.org/pdf/1406.5468.pdf)

[[3] Y. Drori  (2017).
The exact information-based complexity of smooth convex minimization.
Journal of Complexity, 39, 1-16.](https://arxiv.org/pdf/1606.01424.pdf)

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
pepit_tau, theoretical_tau = wc_optimized_gradient(3.0, 4; verbose=true)
```
"""
function wc_optimized_gradient(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    theta_new = 1.0
    x_new = x0
    y = x0
    for i in 0:(n - 1)
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        theta_old = theta_new
        if i < n - 1
            theta_new = (1 + sqrt(4 * theta_new^2 + 1)) / 2
        else
            theta_new = (1 + sqrt(8 * theta_new^2 + 1)) / 2
        end

        y = x_new + (theta_old - 1) / theta_new * (x_new - x_old) + theta_old / theta_new * (x_new - y)
    end


    set_performance_metric!(problem, value!(func, y) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * theta_new^2)


    if verbose
        println("*** Example file: worst-case performance of optimized gradient method ***")
        println("\tPEPit guarantee:\t f(y_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(y_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_optimized_gradient(3.0, 4; verbose=true)
