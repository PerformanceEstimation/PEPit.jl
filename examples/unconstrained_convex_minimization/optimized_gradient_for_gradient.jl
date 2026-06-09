using PEPit, OrderedCollections, Clarabel, OffsetArrays

@doc raw"""
    wc_optimized_gradient_for_gradient(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_optimized_gradient_for_gradient`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code computes a worst-case guarantee for **optimized gradient method for gradient** (OGM-G).
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
\|\nabla f(x_n)\|^2 \leqslant \tau(n, L) (f(x_0) - f_\star)
```

is valid, where $x_n$ is the output of OGM-G and where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$ and $L$, $\tau(n, L)$ is computed as the worst-case value
of $\|\nabla f(x_n)\|^2$ when $f(x_0)-f_\star \leqslant 1$.

# Algorithm

For $t\in\{0,1,\ldots,n-1\}$, the optimized gradient method for gradient [1, Section 6.3] is described by

```math
    \begin{aligned}
        y_{t+1} & = & x_t - \frac{1}{L} \nabla f(x_t),\\
        x_{t+1} & = & y_{t+1} + \frac{(\tilde{\theta}_t-1)(2\tilde{\theta}_{t+1}-1)}{\tilde{\theta}_t(2\tilde{\theta}_t-1)}(y_{t+1}-y_t)+\frac{2\tilde{\theta}_{t+1}-1}{2\tilde{\theta}_t-1}(y_{t+1}-x_t),
    \end{aligned}
```
with

```math
    \begin{aligned}
        \tilde{\theta}_n & = & 1 \\
        \tilde{\theta}_t & = & \frac{1 + \sqrt{4 \tilde{\theta}_{t+1}^2 + 1}}{2}, \forall t \in [|1, n-1|] \\
        \tilde{\theta}_0 & = & \frac{1 + \sqrt{8 \tilde{\theta}_{1}^2 + 1}}{2}.
    \end{aligned}
```
# Theoretical guarantee

The **tight** worst-case guarantee can be found in [1, Theorem 6.1]:

```math
\|\nabla f(x_n)\|^2 \leqslant \frac{2L(f(x_0)-f_\star)}{\tilde{\theta}_0^2},
```

where tightness is achieved on Huber losses, see [1, Section 6.4].

# References

The optimized gradient method for gradient was developed in [1].

[[1] D. Kim, J. Fessler (2021).
Optimizing the efficiency of first-order methods for decreasing the gradient of smooth convex functions.
Journal of optimization theory and applications, 188(1), 192-219.](https://arxiv.org/pdf/1803.06600.pdf)

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
pepit_tau, theoretical_tau = wc_optimized_gradient_for_gradient(3.0, 4; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.307007, 0.307007)
```
"""
function wc_optimized_gradient_for_gradient(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    f0 = value!(func, x0)


    set_initial_condition!(problem, f0 - fs <= 1)


    theta_tmp = [1.0]
    for i in 0:(n - 1)
        if i < n - 1
            push!(theta_tmp, (1 + sqrt(4 * theta_tmp[i + 1]^2 + 1)) / 2)
        else
            push!(theta_tmp, (1 + sqrt(8 * theta_tmp[i + 1]^2 + 1)) / 2)
        end
    end
    reverse!(theta_tmp)
    theta_tilde = OffsetVector(theta_tmp, 0:n)


    x = x0
    y_new = x0

    for i in 0:(n - 1)
        y_old = y_new
        y_new = x - 1 / L * gradient!(func, x)
        x = y_new + (theta_tilde[i] - 1) * (2 * theta_tilde[i + 1] - 1) / theta_tilde[i] / (2 * theta_tilde[i] - 1) *
            (y_new - y_old) + (2 * theta_tilde[i + 1] - 1) / (2 * theta_tilde[i] - 1) * (y_new - x)
    end


    set_performance_metric!(problem, gradient!(func, x)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 * L / (theta_tilde[0]^2)


    if verbose
        println("*** Example file: worst-case performance of optimized gradient method for gradient ***")
        println("\tPEP-it guarantee:\t ||f'(x_n)||^2 <= $(round(pepit_tau, digits=6)) (f(x_0) - f_*)")
        println("\tTheoretical guarantee:\t ||f'(x_n)||^2 <= $(round(theoretical_tau, digits=6)) (f(x_0) - f_*)")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_optimized_gradient_for_gradient(3.0, 4; solver=Clarabel.Optimizer, verbose=true)
