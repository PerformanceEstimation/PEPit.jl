using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_past_extragradient(n, gamma, L; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_past_extragradient`.

Consider the monotone variational inequality

```math
\mathrm{Find}\, x_\star \in C\text{ such that } \left<F(x_\star);x-x_\star\right> \geqslant 0\,\,\forall x\in C,
```

where $C$ is a closed convex set and $F$ is maximally monotone and Lipschitz.

# Performance metric

This code computes a worst-case guarantee for the **past extragradient method**.
That, it computes the smallest possible $\tau(n)$ such that the guarantee

```math
\|x_n - x_{n-1}\|^2 \leqslant \tau(n) \|x_0 - x_\star\|^2,
```

is valid, where $x_n$ is the output of the **past extragradient method** and $x_0$ its starting point.

# Algorithm
The past extragradient method is described as follows, for $t \in \{ 0, \dots, n-1\}$,

```math
    \begin{aligned}
         \tilde{x}_{t} & = & \mathrm{Proj}_{C} [x_t-\gamma F(\tilde{x}_{t-1})], \\
         {x}_{t+1} & = & \mathrm{Proj}_{C} [x_t-\gamma F(\tilde{x}_{t})].
    \end{aligned}
```
where $\gamma$ is some step-size.

# Theoretical guarantee
The method and many variants of it are discussed in [1].
A worst-case guarantee in $O(1/n)$ can be found in [2, 3].

# References


[[1] Y.-G. Hsieh, F. Iutzeler, J. Malick, P. Mertikopoulos (2019).
On the convergence of single-call stochastic extra-gradient methods.
Advances in Neural Information Processing Systems, 32:6938-6948, 2019](https://arxiv.org/pdf/1908.08465.pdf)

[[2] E. Gorbunov, A. Taylor, G. Gidel (2022).
Last-Iterate Convergence of Optimistic Gradient Method for Monotone Variational Inequalities.](https://arxiv.org/pdf/2205.08446.pdf)

[[3] Y. Cai, A. Oikonomou, W. Zheng (2022).
Tight Last-Iterate Convergence of the Extragradient and the Optimistic Gradient Descent-Ascent Algorithm
for Constrained Monotone Variational Inequalities.](https://arxiv.org/pdf/2204.09228.pdf)

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
wc_past_extragradient(5, 1 / 4, 1; verbose=true)
```
"""
function wc_past_extragradient(n, gamma, L; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    ind_C = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    F = declare_function!(problem, LipschitzStronglyMonotoneOperatorCheap, OrderedDict("mu" => 0, "L" => L))

    total_problem = F + ind_C


    xs = stationary_point!(total_problem)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x, _, _ = proximal_step!(x0, ind_C, gamma)
    xtilde = x
    V = gradient!(F, xtilde)
    previous_x = x
    for _ in 1:n
        xtilde, _, _ = proximal_step!(x - gamma * V, ind_C, gamma)
        V = gradient!(F, xtilde)
        previous_x = x
        x, _, _ = proximal_step!(x - gamma * V, ind_C, gamma)
    end


    set_performance_metric!(problem, (x - previous_x)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case performance of the Past Extragradient Method***")
        println("\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_past_extragradient(5, 1 / 4, 1; verbose=true)
