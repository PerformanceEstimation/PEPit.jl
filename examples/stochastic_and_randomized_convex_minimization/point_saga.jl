using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_point_saga(L, mu, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_point_saga`.

Consider the finite sum minimization problem

```math
F^\star \triangleq \min_x \left\{F(x) \equiv \frac{1}{n} \sum_{i=1}^n f_i(x)\right\},
```

where $f_1, \dots, f_n$ are $L$-smooth and $\mu$-strongly convex, and with proximal operator
readily available. In the sequel, we use the notation $\mathbb{E}$ for denoting the expectation over the
uniform distribution of the index $i \sim \mathcal{U}\left([|1, n|]\right)$,
e.g., $F(x)\equiv \mathbb{E}[f_i(x)]$.

# Performance metric

This code computes a tight (one-step) worst-case guarantee using a Lyapunov function for **Point SAGA** [1].
The Lyapunov (or energy) function at a point $x$ is given in [1, Theorem 5]:

```math
V(x) = \frac{1}{L \mu}\frac{1}{n} \sum_{i \leq n} \|\nabla f_i(x) - \nabla f_i(x_\star)\|^2 + \|x - x^\star\|^2,
```

where $x^\star$ denotes the minimizer of $F$. The code computes the smallest possible
$\tau(n, L, \mu)$ such that the guarantee (in expectation):

```math
\mathbb{E}\left[V\left(x^{(1)}\right)\right] \leqslant \tau(n, L, \mu) V\left(x^{(0)}\right),
```

is valid (note that we use the notation $x^{(0)},x^{(1)}$ to denote two consecutive iterates for convenience;
as the bound is valid for all $x^{(0)}$,
it is also valid for any pair of consecutive iterates of the algorithm).

In short, for given values of $n$, $L$, and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$\mathbb{E}\left[V\left(x^{(1)}\right)\right]$ when $V\left(x^{(0)}\right) \leqslant 1$.

# Algorithm

Point SAGA is described by

```math
\begin{aligned}
    \text{Set }\gamma & = & \frac{\sqrt{(n - 1)^2 + 4n\frac{L}{\mu}}}{2Ln} - \frac{\left(1 - \frac{1}{n}\right)}{2L} \\
    \text{Pick random }j & \sim & \mathcal{U}\left([|1, n|]\right) \\
    z^{(t)} & = & x_t + \gamma \left(g_j^{(t)} - \frac{1}{n} \sum_{i\leq n}g_i^{(t)} \right), \\
    x^{(t+1)} & = & \mathrm{prox}_{\gamma f_j}(z^{(t)})\triangleq \arg\min_x\left\{ \gamma f_j(x)+\frac{1}{2} \|x-z^{(t)}\|^2 \right\}, \\
    g_j^{(t+1)} & = & \frac{1}{\gamma}(z^{(t)} - x^{(t+1)}).
\end{aligned}
```
# Theoretical guarantee
A theoretical **upper** bound is given in [1, Theorem 5].

```math
\mathbb{E}\left[V\left(x^{(t+1)}\right)\right] \leqslant \frac{1}{1 + \mu\gamma} V\left(x^{(t)}\right)
```

# References


[[1] A. Defazio (2016). A simple practical accelerated method for finite sums.
Advances in Neural Information Processing Systems (NIPS), 29, 676-684.](https://proceedings.neurips.cc/paper/2016/file/4f6ffe13a5d75b2d6a3923922b3922e5-Paper.pdf)

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
pepit_tau, theoretical_tau = wc_point_saga(1.0, 0.01, 10; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_point_saga(L, mu, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    fn = [declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu)) for _ in 1:n]
    func = sum(fn) / n


    xs = stationary_point!(func)


    phi = [set_initial_point!(problem) for _ in 1:n]
    x0 = set_initial_point!(problem)


    gamma = sqrt((n - 1)^2 + 4 * n * L / mu) / 2 / L / n - (1 - 1 / n) / 2 / L
    c = 1 / (mu * L)


    init_lyapunov = (xs - x0)^2
    gs = [gradient!(fn[i], xs) for i in 1:n]
    for i in 1:n
        init_lyapunov = init_lyapunov + c / n * (gs[i] - phi[i])^2
    end


    set_initial_condition!(problem, init_lyapunov <= 1.0)


    final_lyapunov_avg = (xs - xs)^2
    for i in 1:n
        w = x0 + gamma * phi[i]
        for j in 1:n
            w = w - gamma / n * phi[j]
        end
        x1, gx1, _ = proximal_step!(w, fn[i], gamma)
        final_lyapunov = (xs - x1)^2
        for j in 1:n
            if i != j
                final_lyapunov = final_lyapunov + c / n * (phi[j] - gs[j])^2
            else
                final_lyapunov = final_lyapunov + c / n * (gs[j] - gx1)^2
            end
        end
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n
    end


    set_performance_metric!(problem, final_lyapunov_avg)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    kappa = mu * gamma / (1 + mu * gamma)
    theoretical_tau = (1 - kappa)


    if verbose
        println("*** Example file: worst-case performance of Point SAGA for a given Lyapunov function ***")
        println("\tPEPit guarantee:\t E[V(x^(1))] <= $(round(pepit_tau, digits=6)) V(x^(0))")
        println("\tTheoretical guarantee:\t E[V(x^(1))] <= $(round(theoretical_tau, digits=6)) V(x^(0))")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_point_saga(1.0, 0.01, 10; solver=Clarabel.Optimizer, verbose=true)
