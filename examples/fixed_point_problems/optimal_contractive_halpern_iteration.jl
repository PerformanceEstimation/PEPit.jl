using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_optimal_contractive_halpern_iteration(n, gamma; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_optimal_contractive_halpern_iteration`.

Consider the fixed point problem

```math
\mathrm{Find}\, x:\, x = Ax,
```

where $A$ is a $1/\gamma$-contractive operator,
i.e. a $L$-Lipschitz operator with $L=1/\gamma$.

# Performance metric

This code computes a worst-case guarantee for the **Optimal Contractive Halpern Iteration**.
That is, it computes the smallest possible $\tau(n, \gamma)$ such that the guarantee

```math
\|x_n - Ax_n\|^2 \leqslant \tau(n, \gamma) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the **Optimal Contractive Halpern iteration**,
and $x_\star$ is the fixed point of $A$. In short, for a given value of $n, \gamma$,
$\tau(n, \gamma)$ is computed as the worst-case value of
$\|x_n - Ax_n\|^2$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
The Optimal Contractive Halpern iteration can be written as

```math
x_{t+1} = \left(1 - \frac{1}{\varphi_{t+1}} \right) Ax_t + \frac{1}{\varphi_{t+1}} x_0.
```

where $\varphi_k = \sum_{i=0}^k \gamma^{2i}$ and $x_0$ is a starting point.

# Theoretical guarantee
A **tight** worst-case guarantee for the Optimal Contractive Halpern iteration
can be found in [1, Corollary 3.3, Theorem 4.1]:

```math
\|x_n - Ax_n\|^2 \leqslant \left(1 + \frac{1}{\gamma}\right)^2 \left( \frac{1}{\sum_{k=0}^n \gamma^k} \right)^2 \|x_0 - x_\star\|^2.
```

# References
The detailed approach and tight bound are available in [1].

[[1] J. Park, E. Ryu (2022).
Exact Optimal Accelerated Complexity for Fixed-Point Iterations.
In 39th International Conference on Machine Learning (ICML).](https://proceedings.mlr.press/v162/park22c/park22c.pdf)

# Arguments
- `n`: number of iterations.
- `gamma`: step-size parameter.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_optimal_contractive_halpern_iteration(10, 1.1; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_optimal_contractive_halpern_iteration(n, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => 1 / gamma)
    A = declare_function!(problem, LipschitzOperator, param; reuse_gradient=true)


    xs, _, _ = fixed_point!(A)
    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        phi = (gamma^(2 * i + 4) - 1) / (gamma^2 - 1)
        x = 1 / phi * x0 + (1 - 1 / phi) * gradient!(A, x)
    end


    set_performance_metric!(problem, (x - gradient!(A, x))^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)
    theoretical_tau = (1 + 1 / gamma)^2 * ((gamma - 1) / (gamma^(n + 1) - 1))^2

    if verbose
        println("*** Example file: worst-case performance of Optimal Contractive Halpern Iterations ***")
        println("\tPEPit guarantee:\t ||xN - AxN||^2 <= $(round(pepit_tau, digits=7)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||xN - AxN||^2 <= $(round(theoretical_tau, digits=7)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_optimal_contractive_halpern_iteration(10, 1.1; solver=Clarabel.Optimizer, verbose=true)
