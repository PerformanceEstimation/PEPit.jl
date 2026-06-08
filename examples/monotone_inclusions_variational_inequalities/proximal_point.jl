using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_proximal_point(alpha::Real, n::Int; solver=Clarabel.Optimizer, verbose::Int=1)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_proximal_point`.

Consider the monotone inclusion problem

```math
\mathrm{Find}\, x:\, 0\in Ax,
```

where $A$ is maximally monotone. We denote $J_A = (I + A)^{-1}$ the resolvent of $A$.

# Performance metric

This code computes a worst-case guarantee for the **proximal point** method.
That, it computes the smallest possible $\tau(n, \alpha)$ such that the guarantee

```math
\|x_n - x_{n-1}\|^2 \leqslant \tau(n, \alpha) \|x_0 - x_\star\|^2,
```

is valid, where $x_\star$ is such that $0 \in Ax_\star$.

# Algorithm
The proximal point algorithm for monotone inclusions is described as follows,
for $t \in \{ 0, \dots, n-1\}$,

```math
x_{t+1} = J_{\alpha A}(x_t),
```

where $\alpha$ is a step-size.

# Theoretical guarantee
A tight theoretical guarantee can be found in [1, section 4].

```math
\|x_n - x_{n-1}\|^2 \leqslant \frac{\left(1 - \frac{1}{n}\right)^{n - 1}}{n} \|x_0 - x_\star\|^2.
```

**Reference**:

[[1] G. Gu, J. Yang (2020). Tight sublinear convergence rate of the proximal point algorithm for maximal
monotone inclusion problem. SIAM Journal on Optimization, 30(3), 1905-1921.](https://epubs.siam.org/doi/pdf/10.1137/19M1299049)

# References
No bibliographic reference was listed in the corresponding Python PEPit example docstring.

# Arguments
- `alpha`: algorithm parameter used in the update rule.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
wc_proximal_point(2.0, 10; verbose=1)
```
"""
function wc_proximal_point(alpha::Real, n::Int; solver=Clarabel.Optimizer, verbose::Int=1)

    problem = PEP()


    A = declare_function!(problem, MonotoneOperator, OrderedDict())


    xs = stationary_point!(A)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    previous_x = x0
    for _ in 1:n
        previous_x = x
        x, _, _ = proximal_step!(previous_x, A, alpha)
    end


    set_performance_metric!(problem, (x - previous_x)^2)


    pepit_verbose = verbose >= 0
    τ_PEPit = solve!(problem, solver=solver, verbose=pepit_verbose)


    τ_theory = (1 - 1 / n)^(n - 1) / n

    if verbose != -1
        @info "*** Example file: worst-case performance of the Proximal Point Method***"
        @info "PEPit guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(τ_PEPit, digits=6)) ||x0 - xs||^2"
        @info "Theoretical guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(τ_theory, digits=6)) ||x0 - xs||^2"
    end

    return τ_PEPit, τ_theory
end

τ_PEPit, τ_theory =
wc_proximal_point(2.0, 10; verbose=1)
