using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_conjugate_gradient(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_conjugate_gradient`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code computes a worst-case guarantee for the **conjugate gradient (CG)** method (with exact span searches).
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L) \|x_0-x_\star\|^2
```

is valid, where $x_n$ is the output of the **conjugate gradient** method,
and where $x_\star$ is a minimizer of $f$.
In short, for given values of $n$ and $L$,
$\tau(n, L)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $\|x_0-x_\star\|^2 \leqslant 1$.

# Algorithm


```math
x_{t+1} = x_t - \sum_{i=0}^t \gamma_i \nabla f(x_i)
```

    with

```math
(\gamma_i)_{i \leqslant t} = \arg\min_{(\gamma_i)_{i \leqslant t}} f \left(x_t - \sum_{i=0}^t \gamma_i \nabla f(x_i) \right)
```

# Theoretical guarantee


    The **tight** guarantee obtained in [1] is

```math
f(x_n) - f_\star \leqslant\frac{L}{2 \theta_n^2}\|x_0-x_\star\|^2.
```

    where

```math
    \begin{aligned}
        \theta_0 & = & 1 \\
        \theta_t & = & \frac{1 + \sqrt{4 \theta_{t-1}^2 + 1}}{2}, \forall t \in [|1, n-1|] \\
        \theta_n & = & \frac{1 + \sqrt{8 \theta_{n-1}^2 + 1}}{2},
    \end{aligned}

and tightness follows from [2, Theorem 3].
```
# References

The detailed approach (based on convex relaxations) is available in [1, Corollary 6].

[[1] Y. Drori and A. Taylor (2020).
Efficient first-order methods for convex minimization: a constructive approach.
Mathematical Programming 184 (1), 183-220.](https://arxiv.org/pdf/1803.05676.pdf)

[[2] Y. Drori  (2017).
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
pepit_tau, theoretical_tau = wc_conjugate_gradient(1.0, 2; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.061894, 0.061894)
```
"""
function wc_conjugate_gradient(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    g0, f0 = oracle!(func, x0)
    span = Any[g0]
    local gx, fx
    for i in 1:n
        x_old = x_new
        x_new, gx, fx = exact_linesearch_step!(x_new, func, span)
        push!(span, gx)
        push!(span, x_old - x_new)
    end


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theta_new = 1.0
    for i in 0:(n - 1)
        if i < n - 1
            theta_new = (1 + sqrt(4 * theta_new^2 + 1)) / 2
        else
            theta_new = (1 + sqrt(8 * theta_new^2 + 1)) / 2
        end
    end
    theoretical_tau = L / (2 * theta_new^2)

    if verbose
        println("*** Example file: worst-case performance of conjugate gradient method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_conjugate_gradient(1.0, 2; solver=Clarabel.Optimizer, verbose=true)
