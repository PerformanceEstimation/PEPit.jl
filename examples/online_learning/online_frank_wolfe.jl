using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_online_frank_wolfe(M::Real, D::Real, n::Int; solver = Clarabel.Optimizer, verbose = true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_online_frank_wolfe`.

Consider the online convex minimization problem, whose goal is to sequentially minimize the regret

```math
R_n \triangleq \max_{x\in Q} \sum_{i=1}^n f_i(x_i)-f_i(x),
```

where the functions $f_i$ are $M$-Lipschitz and convex, and where $Q$ is a
bounded closed convex set with diameter upper bounded by $D$. We also denote by $x_\star\in Q$
the solution to the minimization problem defining $R_n$ (i.e., $x_\star$ is a reference point).
Classical references on the topic include [1, 2].

# Performance metric

This code computes a worst-case guarantee for **online Frank-Wolfe** (OFW), see [1, Algorithm 27];
the code uses the choice [3, Section 2] here. That is, it computes the smallest possible
$\tau(n, M, D)$ such that the guarantee

```math
R_n \leqslant \tau(n, M, D)
```

is valid for any such sequence of queries of OFW; that is, $x_t$ are the query points of OFW.

In short, for given values of $n$, $M$, $D$:
$\tau(n, M, D)$ is computed as the worst-case value of $R_n$.

# Algorithm

Online Frank-Wolfe is described by

```math
    \begin{aligned}
        \text{dir}_t & = & x_t-x_1 + \eta \sum_{s=1}^t g_s \\
        v_{t} & = & \text{argmin}_{v\in Q} \langle \text{dir}_t;v\rangle\\
        x_{t+1} & = & (1-\sigma) x_t + \sigma v_t
    \end{aligned}
```
where $\eta=\tfrac{D}{2M}\left(\frac{3}{n} \right)^{3/4}$ and $\sigma=\min\{1,\sqrt{3/n}\}$.

# Theoretical guarantee

We compare the numerical results with those of [3, Theorem 2.1]:

```math
R_n \leqslant \frac{4}{3^{3/4}} MDn^{3/4}
```


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
pepit_tau, theoretical_tau = wc_online_frank_wolfe(M, D, n; verbose=true)
```
"""
function wc_online_frank_wolfe(M::Real, D::Real, n::Int; solver = Clarabel.Optimizer, verbose = true)


    problem = PEP()


    eta = D / (2 * M) * (3 / n)^(3 / 4)
    sigma = min(1, sqrt(3 / n))


    fis = [declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M)) for _ in 1:n]


    h = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))


    x_ref = set_initial_point!(problem)
    x_ref, _, _ = proximal_step!(x_ref, h, 1)


    x1 = set_initial_point!(problem)
    x1, _, _ = proximal_step!(x1, h, 1)


    x = x1
    acc_g = 0 * x_ref
    regret = 0 * x_ref^2

    for i in 1:n
        g, f = oracle!(fis[i], x)
        regret = regret + f - value!(fis[i], x_ref)
        acc_g = acc_g + g
        dir_t = (x - x1) + eta * acc_g
        v, _, _ = linear_optimization_step!(dir_t, h)
        x = (1 - sigma) * x + sigma * v
    end


    set_performance_metric!(problem, regret)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 4 / 3^(3 / 4) * M * D * n^(3 / 4)


    if verbose != -1
        println("*** Example file: worst-case regret of online Frank-Wolfe ***")
        println("\tPEPit guarantee:\t R_n <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t R_n <= $(round(theoretical_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


M, D, n = 1.0, 0.5, 2

pepit_tau, theoretical_tau = wc_online_frank_wolfe(M, D, n; verbose=true)
