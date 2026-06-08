using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_alternate_projections(n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_alternate_projections`.

Consider the convex feasibility problem:

```math
\mathrm{Find}\, x\in Q_1\cap Q_2
```

where $Q_1$ and $Q_2$ are two closed convex sets.

# Performance metric

This code computes a worst-case guarantee for the **alternate projection method**, and looks for a low-dimensional
worst-case example nearly achieving this worst-case guarantee.
That is, it computes the smallest possible $\tau(n)$ such that the guarantee

```math
\|\mathrm{Proj}_{Q_1}(x_n)-\mathrm{Proj}_{Q_2}(x_n)\|^2 \leqslant \tau(n) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the **alternate projection method**,
and $x_\star\in Q_1\cap Q_2$ is a solution to the convex feasibility problem.

In short, for a given value of $n$,
$\tau(n)$ is computed as the worst-case value of
$\|\mathrm{Proj}_{Q_1}(x_n)-\mathrm{Proj}_{Q_2}(x_n)\|^2$
when $\|x_0 - x_\star\|^2 \leqslant 1$.
Then, it looks for a low-dimensional nearly achieving this performance.

# Algorithm
The alternate projection method can be written as

```math
    \begin{aligned}
        y_{t+1} & = & \mathrm{Proj}_{Q_1}(x_t), \\
        x_{t+1} & = & \mathrm{Proj}_{Q_2}(y_{t+1}).
    \end{aligned}
```
# References
The first results on this method are due to [1]. Its translation in PEPs is due to [2].

[[1] J. Von Neumann (1949). On rings of operators. Reduction theory. Annals of Mathematics, pp. 401-485.](https://www.jstor.org/stable/1969463)

[[2] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex optimization.
SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

# Arguments
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: no theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_alternate_projections(10; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_alternate_projections(n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    ind_Q1 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    ind_Q2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    func = ind_Q1 + ind_Q2


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    x = x0
    for _ in 1:n
        y, _, _ = proximal_step!(x, ind_Q1, 1)
        x, _, _ = proximal_step!(y, ind_Q2, 1)
    end


    proj1_x, _, _ = proximal_step!(x, ind_Q1, 1)
    proj2_x = x
    set_performance_metric!(problem, (proj2_x - proj1_x)^2)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=1)
    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case performance of the alternate projection method ***")
        println("\tPEPit guarantee:\t ||Proj_Q1 (xn) - Proj_Q2 (xn)||^2 == $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_alternate_projections(10; solver=Clarabel.Optimizer, verbose=true)
