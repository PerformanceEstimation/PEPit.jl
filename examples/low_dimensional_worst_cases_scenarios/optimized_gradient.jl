using PEPit
using OrderedCollections

@doc raw"""
    wc_optimized_gradient(L, n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_optimized_gradient`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code computes a worst-case guarantee for **optimized gradient method** (OGM), and applies the trace heuristic
for trying to find a low-dimensional worst-case example on which this guarantee is nearly achieved.
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of OGM and where $x_\star$ is a minimizer of $f$.
Then, it applies the trace heuristic, which allows obtaining a one-dimensional function
on which the guarantee is nearly achieved.

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
f(x_n)-f_\star \leqslant \frac{L\|x_0-x_\star\|^2}{2\theta_n^2}.
```

# References
The OGM was developed in [1,2].
Low-dimensional worst-case functions for OGM were obtained in [3, 4].

[[1] Y. Drori, M. Teboulle (2014). Performance of first-order methods for smooth convex minimization: a novel
approach. Mathematical Programming 145(1-2), 451-482.](https://arxiv.org/pdf/1206.3209.pdf)

[[2] D. Kim, J. Fessler (2016). Optimized first-order methods for smooth convex minimization. Mathematical
Programming 159.1-2: 81-107.](https://arxiv.org/pdf/1406.5468.pdf)

[[3] A. Taylor, J. Hendrickx, F. Glineur (2017). Smooth strongly convex interpolation and exact worst-case
performance of first-order methods. Mathematical Programming, 161(1-2), 307-345.](https://arxiv.org/pdf/1502.05666.pdf)

[[4] D. Kim, J. Fessler (2017). On the convergence analysis of the optimized gradient method. Journal of
Optimization Theory and Applications, 172(1), 187-205.](https://arxiv.org/pdf/1510.08573.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_optimized_gradient(3.0, 4; verbose=true)
```
"""
function wc_optimized_gradient(L, n; verbose=true)
    problem = PEP()


    func = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    theta_new = 1.0
    x_new = x0
    y = x0
    for i in 1:n
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        theta_old = theta_new
        if i < n
            theta_new = (1 + sqrt(4 * theta_new^2 + 1)) / 2
        else
            theta_new = (1 + sqrt(8 * theta_new^2 + 1)) / 2
        end
        y = x_new + (theta_old - 1) / theta_new * (x_new - x_old) + theta_old / theta_new * (x_new - y)
    end


    set_performance_metric!(problem, value!(func, y) - fs)

    pepit_tau_orig = solve!(problem; verbose=verbose, tracetrick=false)

    pepit_tau_trace_tricked = solve!(problem; verbose=verbose, tracetrick=true)

    theoretical_tau = L / (2 * theta_new^2)

    if verbose
        println("*** Example file: worst-case performance of optimized gradient method to see application of trace trick***")
        println("\tPEPit guarantee original:\t f(y_n)-f_* == $(round(pepit_tau_orig, digits=8)) ||x_0 - x_*||^2")
        println("\tPEPit guarantee after applying trace trick:\t f(y_n)-f_* == $(round(pepit_tau_trace_tricked, digits=8)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(y_n)-f_* <= $(round(theoretical_tau, digits=8)) ||x_0 - x_*||^2")
    end

    return pepit_tau_orig, theoretical_tau
end


pepit_tau, theoretical_tau = wc_optimized_gradient(3.0, 4; verbose=true)
