using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, d; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_randomized_coordinate_descent_smooth_strongly_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for **randomized block-coordinate descent**
with step-size $\gamma$.
That is, it computes the smallest possible $\tau(L, \mu, \gamma, d)$ such that the guarantee

```math
\mathbb{E}[\|x_{t+1} - x_\star \|^2] \leqslant \tau(L, \mu, \gamma, d) \|x_t - x_\star\|^2
```

holds for any fixed step-size $\gamma$ and any number of blocks $d$,
and where $x_\star$ denotes a minimizer of $f$. The notation $\mathbb{E}$
denotes the expectation over the uniform distribution of the index
$i \sim \mathcal{U}\left([|1, n|]\right)$.

In short, for given values of $\mu$, $L$, $d$, and $\gamma$,
$\tau(L, \mu, \gamma, d)$ is computed as the worst-case value of
$\mathbb{E}[\|x_{t+1} - x_\star \|^2]$ when $\|x_t - x_\star\|^2 \leqslant 1$.

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

When $\gamma \leqslant \frac{1}{L}$, the **tight** theoretical guarantee
can be found in [1, Appendix I, Theorem 17]:

```math
\mathbb{E}[\|x_{t+1} - x_\star \|^2] \leqslant \rho^2 \|x_t-x_\star\|^2,
```

where $\rho^2 = \max \left( \frac{(\gamma\mu - 1)^2 + d - 1}{d},\frac{(\gamma L - 1)^2 + d - 1}{d} \right)$.

# References


[[1] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided
analyses via potential functions. In Conference on Learning Theory (COLT).](https://arxiv.org/pdf/1902.00947.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `d`: the dimension.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, 2; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.834711, 0.834711)
```
"""
function wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, d; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    partition = declare_block_partition!(problem, d)


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    g0 = gradient!(func, x0)
    x1_list = [x0 - gamma * get_block(partition, g0, i) for i in 1:d]


    set_performance_metric!(problem, sum((x1 - xs)^2 for x1 in x1_list) / d)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max(((mu * gamma - 1)^2 + d - 1) / d, ((L * gamma - 1)^2 + d - 1) / d)


    if verbose
        println("*** Example file: worst-case performance of randomized coordinate gradient descent ***")
        println("\tPEPit guarantee:\t E[||x_(t+1) - x_*||^2] <= $(round(pepit_tau, digits=6)) ||x_t - x_*||^2")
        println("\tTheoretical guarantee:\t E[||x_(t+1) - x_*||^2] <= $(round(theoretical_tau, digits=6)) ||x_t - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
mu = 0.1
gamma = 2 / (mu + L)
pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, 2; solver=Clarabel.Optimizer, verbose=true)
