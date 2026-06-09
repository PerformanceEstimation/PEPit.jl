using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_krasnoselskii_mann_increasing_step_sizes(n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_krasnoselskii_mann_increasing_step_sizes`.

Consider the fixed point problem

```math
\mathrm{Find}\, x:\, x = Ax,
```

where $A$ is a non-expansive operator, that is a $L$-Lipschitz operator with $L=1$.

# Performance metric

This code computes a worst-case guarantee for the **Krasnolselskii-Mann** method. That is, it computes
the smallest possible $\tau(n)$ such that the guarantee

```math
\frac{1}{4}\|x_n - Ax_n\|^2 \leqslant \tau(n) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the KM method, and $x_\star$ is some fixed point of $A$
(i.e., $x_\star=Ax_\star$).

# Algorithm
The KM method is described by

```math
x_{t+1} = \frac{1}{t + 2} x_{t} + \left(1 - \frac{1}{t + 2}\right) Ax_{t}.
```

# References

This scheme was first studied using PEPs in [1].

[[1] F. Lieder (2018).
Projection Based Methods for Conic Linear Programming — Optimal First Order
Complexities and Norm Constrained Quasi Newton Methods.
PhD thesis, HHU Düsseldorf.](https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-49971/Dissertation.pdf)

# Arguments
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: no theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_krasnoselskii_mann_increasing_step_sizes(3; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.119634, nothing)
```
"""
function wc_krasnoselskii_mann_increasing_step_sizes(n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => 1.0)
    A = declare_function!(problem, LipschitzOperator, param)


    xs, _, _ = fixed_point!(A)
    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        x = 1 / (i + 2) * x + (1 - 1 / (i + 2)) * gradient!(A, x)
    end


    set_performance_metric!(problem, (1 / 2 * (x - gradient!(A, x)))^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing

    if verbose
        println("*** Example file: worst-case performance of Kranoselskii-Mann iterations ***")
        println("\tPEPit guarantee:\t 1/4 ||xN - AxN||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_krasnoselskii_mann_increasing_step_sizes(3; verbose=true)
