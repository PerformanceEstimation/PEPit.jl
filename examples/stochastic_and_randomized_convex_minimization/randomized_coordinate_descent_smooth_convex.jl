using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_randomized_coordinate_descent_smooth_convex(L, gamma, d, t; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_randomized_coordinate_descent_smooth_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code computes a worst-case guarantee for **randomized block-coordinate descent** with $d$ blocks and
fixed step-size $\gamma$.
That is, it verifies that the Lyapunov function

```math
\phi(t, x_t) = (t \gamma \frac{L}{d} + 1)(f(x_t) - f_\star) + \frac{L}{2} \|x_t - x_\star\||^2
```

is decreasing in expectation over the **randomized block-coordinate descent** algorithm. We use the notation
$\mathbb{E}$ for denoting the expectation over the uniform distribution
of the index $i \sim \mathcal{U}\left([|1, n|]\right)$.

In short, for given values of $L$, $d$, and $\gamma$, it computes the worst-case value
of $\mathbb{E}[\phi(t, x_t)]$ such that $\phi(x_{t-1}) \leqslant 1$.

# Algorithm

Randomized block-coordinate descent is described by

```math
\begin{aligned}
    \text{Pick random }i & \sim & \mathcal{U}\left([|1, d|]\right), \\
    x_{t+1} & = & x_t - \gamma \nabla_i f(x_t),
\end{aligned}
```
where $\gamma$ is a step-size and $\nabla_i f(x_t)$ is the $i^{\text{th}}$ partial gradient.

# Theoretical guarantee

When $\gamma \leqslant \frac{1}{L}$,
the **tight** theoretical guarantee can be found in [1, Appendix I, Theorem 16]:

```math
\mathbb{E}[\phi(t, x_t)] \leqslant \phi(t-1, x_{t-1}),
```

where $\phi(t, x_t) = (t \gamma \frac{L}{d} + 1)(f(x_t) - f_\star) + \frac{L}{2} \|x_t - x_\star\|^2$.

# References


[[1] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided
analyses via potential functions. In Conference on Learning Theory (COLT).](https://arxiv.org/pdf/1902.00947.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `d`: the dimension.
- `t`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_convex(L, 1 / L, 2, 4; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (1.0, 1.0)
```
"""
function wc_randomized_coordinate_descent_smooth_convex(L, gamma, d, t; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    partition = declare_block_partition!(problem, d)


    func = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt_minus_1 = set_initial_point!(problem)


    phi(k, x) = (k * gamma * L / d + 1) * (value!(func, x) - fs) + L / 2 * (x - xs)^2


    set_initial_condition!(problem, phi(t - 1, xt_minus_1) <= 1)


    gt_minus_1 = gradient!(func, xt_minus_1)
    xt_list = [xt_minus_1 - gamma * get_block(partition, gt_minus_1, i) for i in 1:d]


    phi_t = sum(phi(t, xt) for xt in xt_list) / d


    set_performance_metric!(problem, phi_t)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 1.0


    if verbose
        println("*** Example file: worst-case performance of randomized  coordinate gradient descent ***")
        println("\tPEPit guarantee:\t E[phi(t, x_t)] <= $(round(pepit_tau, digits=6)) phi(t-1, x_(t-1))")
        println("\tTheoretical guarantee:\t E[phi(t, x_t)] <= $(round(theoretical_tau, digits=6)) phi(t-1, x_(t-1))")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_convex(L, 1 / L, 2, 4; solver=Clarabel.Optimizer, verbose=true)
