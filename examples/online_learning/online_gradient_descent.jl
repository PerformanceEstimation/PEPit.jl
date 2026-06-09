using PEPit, OrderedCollections

@doc raw"""
    wc_online_gradient_descent(M::Real, D::Real, n::Int; verbose = true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_online_gradient_descent`.

Consider the online convex minimization problem, whose goal is to sequentially minimize the regret

```math
R_n \triangleq \max_{x\in Q} \sum_{i=1}^n f_i(x_i)-f_i(x),
```

where the functions $f_i$ are $M$-Lipschitz and convex, and where $Q$ is a
bounded closed convex set with diameter upper bounded by $D$. We also denote by $x_\star\in Q$
the solution to the minimization problem defining $R_n$ (i.e., $x_\star$ is a reference point).
Classical references on the topic include [1, 2]; such algorithms were studied using the performance
estimation technique in [3] and using the related IQCs in [4].

# Performance metric

This code computes a worst-case guarantee for **online gradient descent** (OGD) with a step-size $\gamma=D/M/\sqrt{n}$.
That is, it computes the smallest possible $\tau(n, M, D)$ such that the guarantee

```math
R_n \leqslant \tau(n, M, D)
```

is valid for any such sequence of queries of OGD; that is, $x_t$ are the query points of OGD.

In short, for given values of $n$, $M$, $D$:
$\tau(n, M, D)$ is computed as the worst-case value of $R_n$.

# Algorithm

Online gradient descent is described by

```math
x_{t+1} = x_t - \gamma \nabla f_t(x_t),
```

where $\gamma=D/M/\sqrt{n}$ is a step-size.

# Theoretical guarantee

We compare the numerical results with those of [2, Section 2.1.2]:

```math
R_n \leqslant MD\sqrt{n}.
```


# References


[[1] E. Hazan (2016).
Introduction to online convex optimization.
Foundations and Trends in Optimization, 2(3-4), 157-325.](https://arxiv.org/pdf/1912.13213)

[[2] F. Orabona (2025).
A Modern Introduction to Online Learning.](https://arxiv.org/pdf/1912.13213)

[[3] J. Weibel, P. Gaillard, W.M. Koolen, A. Taylor (2025).
Optimized projection-free algorithms for online learning: construction and worst-case analysis](https://arxiv.org/pdf/2506.05855)

[[4] F. Jakob, A. Iannelli (2025).
Online Convex Optimization and Integral Quadratic Constraints: A new approach to regret analysis](https://arxiv.org/pdf/2503.23600?)

# Arguments
- `M`: the Lipschitz parameter.
- `D`: the diameter of the set.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_online_gradient_descent(M, D, n; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.707107, 0.707107)
```
"""
function wc_online_gradient_descent(M::Real, D::Real, n::Int; verbose = true)


    problem = PEP()


    gamma = D / (M * sqrt(n))


    fis = [declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M)) for _ in 1:n]


    h = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))


    F = sum(fis)


    x_ref = set_initial_point!(problem)
    x_ref, _, _ = proximal_step!(x_ref, h, 1)
    _, F_ref = oracle!(F, x_ref)


    x = set_initial_point!(problem)
    x, _, _ = proximal_step!(x, h, 1)


    f_saved = Vector{Expression}(undef, n)
    for i in 1:n
        g_i, f_i = oracle!(fis[i], x)
        f_saved[i] = f_i
        x, _, _ = proximal_step!(x - gamma * g_i, h, gamma)
    end


    set_performance_metric!(problem, sum(f_saved) - F_ref)


    pepit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = M * D * sqrt(n)


    if verbose != -1
        println("*** Example file: worst-case regret of online gradient descent for fixed step-sizes ***")
        println("\tPEPit guarantee:\t R_n <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t R_n <= $(round(theoretical_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


M, D, n = 1.0, 0.5, 2

pepit_tau, theoretical_tau = wc_online_gradient_descent(M, D, n; verbose=true)
