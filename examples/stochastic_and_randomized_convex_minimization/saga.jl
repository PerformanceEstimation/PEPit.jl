using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_saga(L, mu, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_saga`.

Consider the finite sum convex minimization problem

```math
F_\star \triangleq \min_x \left\{F(x) \equiv h(x) + \frac{1}{n} \sum_{i=1}^{n} f_i(x)\right\},
```

where the functions $f_i$ are assumed to be $L$-smooth $\mu$-strongly convex, and $h$ is
closed, proper, and convex with a proximal operator readily available. In the sequel, we use the notation
$\mathbb{E}$ for denoting the expectation over the uniform distribution of the index
$i \sim \mathcal{U}\left([|1, n|]\right)$,
e.g., $F(x)\equiv h(x)+\mathbb{E}[f_i(x)]$.

# Performance metric

This code computes the exact rate for a Lyapunov (or energy) function for **SAGA** [1].
That is, it computes the smallest possible $\tau(n,L,\mu)$ such this Lyapunov function decreases geometrically

```math
\mathbb{E}[V^{(1)}] \leqslant \tau(n, L, \mu) V^{(0)},
```

where the value of the Lyapunov function at iteration $t$ is denoted by $V^{(t)}$ and is defined as

```math
V^{(t)} \triangleq \frac{1}{n} \sum_{i=1}^n \left(f_i(\phi_i^{(t)}) - f_i(x^\star) - \langle \nabla f_i(x^\star); \phi_i^{(t)} - x^\star\rangle\right) + \frac{1}{2 n \gamma (1-\mu \gamma)} \|x^{(t)} - x^\star\|^2,
```

with $\gamma = \frac{1}{2(\mu n+L)}$ (this Lyapunov function was proposed in [1, Theorem 1]).
We consider the case $t=0$ in the code below, without loss of generality.

In short, for given values of $n$, $L$, and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of $\mathbb{E}[V^{(1)}]$
when $V(x^{(0)}) \leqslant 1$.

# Algorithm
One iteration of SAGA [1] is described as follows: at iteration $t$, pick
$j\in\{1,\ldots,n\}$ uniformely at random and set:

```math
    \begin{aligned}
        \phi_j^{(t+1)} & = & x^{(t)} \\
        w^{(t+1)} & = & x^{(t)} - \gamma \left[ \nabla f_j (\phi_j^{(t+1)}) - \nabla f_j(\phi_j^{(t)}) + \frac{1}{n} \sum_{i=1}^n(\nabla f_i(\phi^{(t)}))\right] \\
        x^{(t+1)} & = & \mathrm{prox}_{\gamma h} (w^{(t+1)})\triangleq \arg\min_x \left\{ \gamma h(x)+\frac{1}{2}\|x-w^{(t+1)}\|^2\right\}
    \end{aligned}
```
# Theoretical guarantee
The following **upper** bound (empirically tight) can be found in [1, Theorem 1]:

```math
\mathbb{E}[V^{(t+1)}] \leqslant \left(1-\gamma\mu \right)V^{(t)}
```

# References


[[1] A. Defazio, F. Bach, S. Lacoste-Julien (2014). SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives.
In Advances in Neural Information Processing Systems (NIPS).](http://papers.nips.cc/paper/2014/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_saga(1.0, 0.1, 5; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_saga(L, mu, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    h = declare_function!(problem, ConvexFunction, OrderedDict())
    fn = [declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu); reuse_gradient=true) for _ in 1:n]


    func = h + sum(fn) / n


    xs = stationary_point!(func)


    phi = [set_initial_point!(problem) for _ in 1:n]
    x0 = set_initial_point!(problem)


    gamma = 1 / 2 / (mu * n + L)
    c = 1 / 2 / gamma / (1 - mu * gamma) / n
    g, f = Vector{Any}(undef, n), Vector{Any}(undef, n)
    g0, f0 = Vector{Any}(undef, n), Vector{Any}(undef, n)
    gs, fs = Vector{Any}(undef, n), Vector{Any}(undef, n)
    init_lyapunov = c * (xs - x0)^2

    for i in 1:n
        g[i], f[i] = oracle!(fn[i], phi[i])
        gs[i], fs[i] = oracle!(fn[i], xs)
        init_lyapunov = init_lyapunov + 1 / n * (f[i] - fs[i] - gs[i] * (phi[i] - xs))
    end


    set_initial_condition!(problem, init_lyapunov <= 1)


    final_lyapunov_avg = (xs - xs)^2
    for i in 1:n
        g0[i], f0[i] = oracle!(fn[i], x0)
        w = x0 - gamma * (g0[i] - g[i])
        for j in 1:n
            w = w - gamma / n * g[j]
        end
        x1, _, _ = proximal_step!(w, h, gamma)
        final_lyapunov = c * (x1 - xs)^2
        for j in 1:n
            if i != j
                gi, fi = g[j], f[j]
                final_lyapunov = final_lyapunov + 1 / n * (fi - fs[j] - gs[j] * (phi[j] - xs))
            else
                gi, fi = g0[i], f0[i]
                final_lyapunov = final_lyapunov + 1 / n * (fi - fs[j] - gs[j] * (x0 - xs))
            end
        end
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n
    end


    set_performance_metric!(problem, final_lyapunov_avg)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 - gamma * mu)


    if verbose
        println("*** Example file: worst-case performance of SAGA for Lyapunov function V_t ***")
        println("\tPEPit guarantee:\t V^(1) <= $(round(pepit_tau, digits=6)) V^(0)")
        println("\tTheoretical guarantee:\t V^(1) <= $(round(theoretical_tau, digits=6)) V^(0)")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_saga(1.0, 0.1, 5; solver=Clarabel.Optimizer, verbose=true)
