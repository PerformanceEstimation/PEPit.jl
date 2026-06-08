using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_online_follow_regularized_leader(M::Real, D::Real, n::Int; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_online_follow_regularized_leader`.

Consider the online convex minimization problem, whose goal is to sequentially minimize the regret

```math
R_n \triangleq \max_{x\in Q} \sum_{i=1}^n f_i(x_i)-f_i(x),
```

where the functions $f_i$ are $M$-Lipschitz and convex, and where $Q$ is a
bounded closed convex set with diameter upper bounded by $D$. We also denote by $x_\star\in Q$
the solution to the minimization problem defining $R_n$ (i.e., $x_\star$ is a reference point).
Classical references on the topic include [1, 2]; such algorithms were studied using the performance
estimation technique in [3].

# Performance metric

This code computes a worst-case guarantee for **follow the regularized leader** (FTRL).
That is, it computes the smallest possible $\tau(n, M, D)$ such that the guarantee

```math
R_n \leqslant \tau(n, M, D)
```

is valid for any such sequence of queries of FTRL; that is, $x_t$ are the query points of OGD.

In short, for given values of $n$, $M$, $D$:
$\tau(n, M, D)$ is computed as the worst-case value of $R_n$.

# Algorithm

Follow the regularized leader is described by

```math
x_{t+1} \in \text{argmin}_{x\in Q} \left( \sum_{i=1}^t f_i(x) + \tfrac{\eta}{2}\|x-x_1\|^2 \right).
```

# Theoretical guarantee
The follow the regularized leader strategy is known to enjoy sublinear regret
(see, e.g., [1, Theorem 5.2]); we compare with the bound:

```math
R_n \leqslant MD\sqrt{n}
```

with a regularization strength $\eta=D/M/\sqrt{n}$.


# References


[[1] E. Hazan (2016).
Introduction to online convex optimization.
Foundations and Trends in Optimization, 2(3-4), 157-325.](https://arxiv.org/pdf/1912.13213)

[[2] F. Orabona (2025).
A Modern Introduction to Online Learning.](https://arxiv.org/pdf/1912.13213)

[[3] J. Weibel, P. Gaillard, W.M. Koolen, A. Taylor (2025).
Optimized projection-free algorithms for online learning: construction and worst-case analysis](https://arxiv.org/pdf/2506.05855)

# Arguments
- `M`: the Lipschitz parameter.
- `D`: the diameter of the set.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_online_follow_regularized_leader(M, D, n; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_online_follow_regularized_leader(M::Real, D::Real, n::Int; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    M_list = [M for i in 1:n]
    eta = D / (M * sqrt(n))


    fis = [declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M_list[i])) for i in 1:n]


    h = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))


    F = sum(fis)


    x_ref = set_initial_point!(problem)
    x_ref, _, _ = proximal_step!(x_ref, h, 1)
    _, F_ref = oracle!(F, x_ref)


    x1 = set_initial_point!(problem)
    x1, _, _ = proximal_step!(x1, h, 1)


    f_occ = h
    x = x1
    f_saved = Vector{Expression}(undef, n)
    for i in 1:n
        g_i, f_i = oracle!(fis[i], x)
        f_saved[i] = f_i
        f_occ = f_occ + fis[i]
        if i < n
            x, _, _ = proximal_step!(x1, f_occ, eta)
        end
    end


    set_performance_metric!(problem, sum(f_saved) - F_ref)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = M * D * sqrt(n)


    if verbose != -1
        println("*** Example file: worst-case regret of online follow the regularized leader ***")
        println("\tPEPit guarantee:\t R_n <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t R_n <= $(round(theoretical_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


M, D, n = 1.0, 0.5, 2

pepit_tau, theoretical_tau = wc_online_follow_regularized_leader(M, D, n; solver=Clarabel.Optimizer, verbose=true)
