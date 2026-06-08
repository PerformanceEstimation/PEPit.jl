using PEPit, OrderedCollections, Clarabel, OffsetArrays

@doc raw"""
    wc_accelerated_proximal_point(alpha::Real, n::Int; solver=Clarabel.Optimizer, verbose::Int=1)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_proximal_point`.

Consider the monotone inclusion problem

```math
\mathrm{Find}\, x:\, 0\in Ax,
```

where $A$ is maximally monotone. We denote $J_A = (I + A)^{-1}$ the resolvent of $A$.

# Performance metric

This code computes a worst-case guarantee for the **accelerated proximal point** method proposed in [1].
That, it computes the smallest possible $\tau(n, \alpha)$ such that the guarantee

```math
\|x_n - y_n\|^2 \leqslant \tau(n, \alpha) \|x_0 - x_\star\|^2,
```

is valid, where $x_\star$ is such that $0 \in Ax_\star$.

# Algorithm
Accelerated proximal point is described as follows, for $t \in \{ 0, \dots, n-1\}$

```math
    \begin{aligned}
        x_{t+1} & = & J_{\alpha A}(y_t), \\
        y_{t+1} & = & x_{t+1} + \frac{t}{t+2}(x_{t+1} - x_{t}) - \frac{t}{t+2}(x_t - y_{t-1}),
    \end{aligned}
```
where $x_0=y_0=y_{-1}$

# Theoretical guarantee
A tight theoretical worst-case guarantee can be found in [1, Theorem 4.1],
for $n \geqslant 1$,

```math
\|x_n - y_{n-1}\|^2 \leqslant  \frac{1}{n^2}  \|x_0 - x_\star\|^2.
```

**Reference**:

[[1] D. Kim (2021). Accelerated proximal point method for maximally monotone operators.
Mathematical Programming, 1-31.](https://arxiv.org/pdf/1905.05149v4.pdf)

# References
No bibliographic reference was listed in the corresponding Python PEPit example docstring.

# Arguments
- `alpha`: algorithm parameter used in the update rule.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
wc_accelerated_proximal_point(2.0, 10; verbose=1)
```
"""
function wc_accelerated_proximal_point(alpha::Real, n::Int; solver=Clarabel.Optimizer, verbose::Int=1)

    problem = PEP()


    A = declare_function!(problem, MonotoneOperator, OrderedDict())


    xs = stationary_point!(A)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = OffsetVector(fill(x0, n + 1), 0:n)
    y = OffsetVector(fill(x0, n + 1), 0:n)

    for i in 0:(n - 2)
        x[i + 1], _, _ = proximal_step!(y[i + 1], A, alpha)
        y[i + 2] = x[i + 1] + i / (i + 2) * (x[i + 1] - x[i]) - i / (i + 2) * (x[i] - y[i])
    end
    x[n], _, _ = proximal_step!(y[n], A, alpha)


    set_performance_metric!(problem, (x[n] - y[n])^2)


    pepit_verbose = verbose >= 0
    τ_PEPit = solve!(problem, solver=solver, verbose=pepit_verbose)


    τ_theory = 1 / (n^2)

    if verbose != -1
        @info "*** Example file: worst-case performance of the Accelerated Proximal Point Method***"
        @info "PEPit guarantee:\t ||x_n - y_n||^2 <= $(round(τ_PEPit, digits=6)) ||x_0 - x_s||^2"
        @info "Theoretical guarantee:\t ||x_n - y_n||^2 <= $(round(τ_theory, digits=6)) ||x_0 - x_s||^2"
    end

    return τ_PEPit, τ_theory
end

τ_PEPit, τ_theory =
wc_accelerated_proximal_point(2.0, 10; verbose=1)
