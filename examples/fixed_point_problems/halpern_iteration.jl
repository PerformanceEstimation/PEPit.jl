using PEPit
using OrderedCollections

@doc raw"""
    wc_halpern_iteration(n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_halpern_iteration`.

Consider the fixed point problem

```math
\mathrm{Find}\, x:\, x = Ax,
```

where $A$ is a non-expansive operator,
that is a $L$-Lipschitz operator with $L=1$.

# Performance metric

This code computes a worst-case guarantee for the **Halpern Iteration**.
That is, it computes the smallest possible $\tau(n)$ such that the guarantee

```math
\|x_n - Ax_n\|^2 \leqslant \tau(n) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the **Halpern iteration**,
and $x_\star$ the fixed point of $A$.

In short, for a given value of $n$,
$\tau(n)$ is computed as the worst-case value of
$\|x_n - Ax_n\|^2$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
The Halpern iteration can be written as

```math
x_{t+1} = \frac{1}{t + 2} x_0 + \left(1 - \frac{1}{t + 2}\right) Ax_t.
```

# Theoretical guarantee
A **tight** worst-case guarantee for Halpern iteration can be found in [2, Theorem 2.1]:

```math
\|x_n - Ax_n\|^2 \leqslant \left(\frac{2}{n+1}\right)^2 \|x_0 - x_\star\|^2.
```

# References
The method was first proposed in [1]. The detailed analysis and tight bound are available in [2].

[[1] B. Halpern (1967).
Fixed points of nonexpanding maps.
American Mathematical Society, 73(6), 957-961.](https://www.ams.org/journals/bull/1967-73-06/S0002-9904-1967-11864-0/S0002-9904-1967-11864-0.pdf)

[[2] F. Lieder (2021).
On the convergence rate of the Halpern-iteration.
Optimization Letters, 15(2), 405-418.](http://www.optimization-online.org/DB_FILE/2017/11/6336.pdf)

# Arguments
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_halpern_iteration(25; verbose=true)
```
"""
function wc_halpern_iteration(n; verbose=true)
    problem = PEP()


    param = OrderedDict("L" => 1.0)
    A = declare_function!(problem, LipschitzOperator, param; reuse_gradient=true)


    xs, _, _ = fixed_point!(A)
    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        x = 1 / (i + 2) * x0 + (1 - 1 / (i + 2)) * gradient!(A, x)
    end


    set_performance_metric!(problem, (x - gradient!(A, x))^2)

    pepit_tau = solve!(problem; verbose=verbose)
    theoretical_tau = (2 / (n + 1))^2

    if verbose
        println("*** Example file: worst-case performance of Halpern Iterations ***")
        println("\tPEPit guarantee:\t ||xN - AxN||^2 <= $(round(pepit_tau, digits=8)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||xN - AxN||^2 <= $(round(theoretical_tau, digits=8)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_halpern_iteration(25; verbose=true)
