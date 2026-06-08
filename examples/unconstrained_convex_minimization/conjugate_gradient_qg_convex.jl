using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_conjugate_gradient_qg_convex(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_conjugate_gradient_qg_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is quadratically upper bounded ($\text{QG}^+$ [2]), i.e.
$\forall x, f(x) - f_\star \leqslant \frac{L}{2} \|x-x_\star\|^2$, and convex.

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


    The **tight** guarantee obtained in [2, Theorem 2.3] (lower) and [2, Theorem 2.4] (upper) is

```math
f(x_n) - f_\star \leqslant \frac{L}{2 (n + 1)} \|x_0-x_\star\|^2.
```

# References

The detailed approach (based on convex relaxations) is available in [1, Corollary 6],
and the result provided in [2, Theorem 2.4].

[[1] Y. Drori and A. Taylor (2020). Efficient first-order methods for convex minimization: a constructive approach.
Mathematical Programming 184 (1), 183-220.](https://arxiv.org/pdf/1803.05676.pdf)

[[2] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
Optimal first-order methods for convex functions with a quadratic upper bound.](https://arxiv.org/pdf/2205.15033.pdf)

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
pepit_tau, theoretical_tau = wc_conjugate_gradient_qg_convex(1.0, 12; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_conjugate_gradient_qg_convex(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexQGFunction, OrderedDict("L" => L))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    g0, f0 = oracle!(func, x0)
    span = [g0]
    local gx, fx
    for i in 1:n
        x_old = x_new
        x_new, gx, fx = exact_linesearch_step!(x_new, func, span)
        push!(span, gx)
        push!(span, x_old - x_new)
    end


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * (n + 1))

    if verbose
        println("*** Example file: worst-case performance of conjugate gradient method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_conjugate_gradient_qg_convex(1.0, 12; solver=Clarabel.Optimizer, verbose=true)
