using PEPit
using OrderedCollections

@doc raw"""
    wc_inconsistent_halpern_iteration(n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_inconsistent_halpern_iteration`.

Consider the fixed point problem

```math
\mathrm{Find}\, x:\, x = Ax,
```

where $A$ is a non-expansive operator,
that is a $L$-Lipschitz operator with $L=1$.
When the solution of above problem, or fixed point, does not exist,
behavior of the fixed-point iteration with A can be characterized with
infimal displacement vector $v$.

# Performance metric

This code computes a worst-case guarantee for the **Halpern Iteration**,
when `A` is not necessarily consistent, i.e., does not necessarily have fixed point.
That is, it computes the smallest possible $\tau(n)$ such that the guarantee

```math
\|x_n - Ax_n - v\|^2 \leqslant \tau(n) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the **Halpern iteration**
and $x_\star$ is the point where $v$ is attained, i.e.,

```math
v = x_\star - Ax_\star
```

In short, for a given value of $n$,
$\tau(n)$ is computed as the worst-case value of
$\|x_n - Ax_n - v\|^2$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
The Halpern iteration can be written as

```math
x_{t+1} = \frac{1}{t + 2} x_0 + \left(1 - \frac{1}{t + 2}\right) Ax_t.
```

# Theoretical guarantee
A worst-case guarantee for Halpern iteration can be found in [1, Theorem 8]:

```math
\|x_n - Ax_n - v\|^2 \leqslant \left(\frac{\sqrt{Hn + 12} + 1}{n + 1}\right)^2 \|x_0 - x_\star\|^2.
```

# References
The detailed approach is available in [1].

[[1] J. Park, E. Ryu (2023).
Accelerated Infeasibility Detection of Constrained Optimization and Fixed-Point Iterations.
International Conference on Machine Learning.](https://arxiv.org/pdf/2303.15876.pdf)

# Arguments
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_inconsistent_halpern_iteration(25; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.025083, 0.036642)
```
"""
function wc_inconsistent_halpern_iteration(n; verbose=true)
    problem = PEP()


    A = declare_function!(problem, NonexpansiveOperator, OrderedDict(); reuse_gradient=true)


    xs = Point()
    Txs = gradient!(A, xs)
    A.v = xs - Txs


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        x = 1 / (i + 2) * x0 + (1 - 1 / (i + 2)) * gradient!(A, x)
    end


    set_performance_metric!(problem, (x - gradient!(A, x) - A.v)^2)

    pepit_tau = solve!(problem; verbose=verbose)

    Hn = sum(1.0 / k for k in 1:n)
    theoretical_tau = ((sqrt(Hn + 12) + 1) / (n + 1))^2

    if verbose
        println("*** Example file: worst-case performance of (possibly inconsistent) Halpern Iterations ***")
        println("\tPEPit guarantee:\t ||xN - AxN - v||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||xN - AxN - v||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau

end


pepit_tau, theoretical_tau = wc_inconsistent_halpern_iteration(25; verbose=true)
