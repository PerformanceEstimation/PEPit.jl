using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_optimistic_gradient_refined(n::Int, gamma::Real, L::Real; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_optimistic_gradient_refined`.

Consider the monotone variational inequality

```math
\mathrm{Find}\, x_\star \in C\text{ such that } \left<F(x_\star);x-x_\star\right> \geqslant 0\,\,\forall x\in C,
```

where $C$ is a closed convex set and $F$ is maximally monotone and Lipschitz.
In this example, we use the characterization of Lipschitz monotone operators provided in [3, Proposition 3.15]
(which results in more computationnaly expensive PEPs to be solved).

# Performance metric

This code computes a worst-case guarantee for the **optimistic gradient method**.
That, it computes the smallest possible $\tau(n)$ such that the guarantee

```math
\|\tilde{x}_n - \tilde{x}_{n-1}\|^2 \leqslant \tau(n) \|x_0 - x_\star\|^2,
```

is valid, where $\tilde{x}_n$ is the output of the **optimistic gradient method**
and $x_0$ its starting point.

# Algorithm
The optimistic gradient method is described as follows, for $t \in \{ 0, \dots, n-1\}$,

```math
    \begin{aligned}
         \tilde{x}_{t} & = & \mathrm{Proj}_{C} [x_t-\gamma F(\tilde{x}_{t-1})], \\
         {x}_{t+1} & = & \tilde{x}_t + \gamma (F(\tilde{x}_{t-1}) - F(\tilde{x}_t)).
    \end{aligned}
```
where $\gamma$ is some step-size.

# Theoretical guarantee
The method and many variants of it are discussed in [1] and a PEP formulation suggesting
a worst-case guarantee in $O(1/n)$ can be found in [2, Appendix D].

# References


[[1] Y.-G. Hsieh, F. Iutzeler, J. Malick, P. Mertikopoulos (2019).
On the convergence of single-call stochastic extra-gradient methods.
Advances in Neural Information Processing Systems, 32:6938-6948, 2019](https://arxiv.org/pdf/1908.08465.pdf)

[[2] E. Gorbunov, A. Taylor, G. Gidel (2022).
Last-Iterate Convergence of Optimistic Gradient Method for Monotone Variational Inequalities.](https://arxiv.org/pdf/2205.08446.pdf)

[[3] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
A constructive approach to strengthen algebraic descriptions of function and operator classes.](https://arxiv.org/pdf/2504.14377.pdf)

# Arguments
- `n`: number of iterations.
- `gamma`: step-size parameter.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: no theoretical bound.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_optimistic_gradient_refined(1, 1 / 4, 1; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (1.062499, nothing)
```
"""
function wc_optimistic_gradient_refined(n::Int, gamma::Real, L::Real; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    ind_C = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    F = declare_function!(problem, LipschitzStronglyMonotoneOperatorExpensive, OrderedDict("mu" => 0, "L" => L))

    total_problem = F + ind_C


    xs = stationary_point!(total_problem)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x, _, _ = proximal_step!(x0, ind_C, gamma)
    xtilde = x
    V = gradient!(F, xtilde)
    previous_xtilde = xtilde
    for _ in 1:n
        previous_xtilde = xtilde
        xtilde, _, _ = proximal_step!(x - gamma * V, ind_C, gamma)
        previous_V = V
        V = gradient!(F, xtilde)
        x = xtilde + gamma * (previous_V - V)
    end


    set_performance_metric!(problem, (xtilde - previous_xtilde)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case performance of the Optimistic Gradient Method***")
        println("\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_optimistic_gradient_refined(1, 1 / 4, 1; verbose=true)
