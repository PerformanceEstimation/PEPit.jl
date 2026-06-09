using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_polyak_steps_in_distance_to_optimum(L, mu, gamma; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_polyak_steps_in_distance_to_optimum`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex, and $x_\star=\arg\min_x f(x)$.

# Performance metric

This code computes a worst-case guarantee for a variant of a **gradient method** relying on **Polyak step-sizes**
(PS). That is, it computes the smallest possible $\tau(L, \mu, \gamma)$ such that the guarantee

```math
\|x_{t+1} - x_\star\|^2 \leqslant \tau(L, \mu, \gamma) \|x_{t} - x_\star\|^2
```

is valid, where $x_t$ is the output of the gradient method with PS and $\gamma$ is the effective
value of the step-size of the gradient method with PS.

In short, for given values of $L$, $\mu$, and $\gamma$, $\tau(L, \mu, \gamma)$ is
computed as the worst-case value of $\|x_{t+1} - x_\star\|^2$ when
$\|x_{t} - x_\star\|^2 \leqslant 1$.

# Algorithm

Gradient descent is described by

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size. The Polyak step-size rule under consideration here corresponds to choosing
of $\gamma$ satisfying:

```math
\gamma \|\nabla f(x_t)\|^2 = 2 (f(x_t) - f_\star).
```

# Theoretical guarantee
The gradient method with the variant of Polyak step-sizes under consideration enjoys the
**tight** theoretical guarantee [1, Proposition 1]:

```math
\|x_{t+1} - x_\star\|^2 \leqslant \tau(L, \mu, \gamma) \|x_{t} - x_\star\|^2,
```

    where $\gamma$ is the effective step-size used at iteration $t$ and

```math
    \begin{aligned}
        \tau(L, \mu, \gamma) & = & \left\{\begin{array}{ll} \frac{(\gamma L-1)(1-\gamma \mu)}{\gamma(L+\mu)-1}  & \text{if } \gamma\in[\tfrac{1}{L},\tfrac{1}{\mu}],\\
        0 & \text{otherwise.} \end{array}\right.
    \end{aligned}
```
# References


[[1] M. Barre, A. Taylor, A. d'Aspremont (2020).
Complexity guarantees for Polyak steps with momentum.
In Conference on Learning Theory (COLT).](https://arxiv.org/pdf/2002.00915.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_polyak_steps_in_distance_to_optimum(1.0, 0.1, 2 / (1.0 + 0.1); solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.669421, 0.669421)
```
"""
function wc_polyak_steps_in_distance_to_optimum(L, mu, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    g0, f0 = oracle!(func, x0)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x1 = x0 - gamma * g0
    _, _ = oracle!(func, x1)


    add_constraint!(problem, gamma * g0^2 == 2 * (f0 - fs))


    set_performance_metric!(problem, (x1 - xs)^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 / L <= gamma <= 1 / mu) ?
        (gamma * L - 1) * (1 - gamma * mu) / (gamma * (L + mu) - 1) :
        0.0

    if verbose
        println("*** Example file: worst-case performance of Polyak steps ***")
        println("\tPEPit guarantee:\t ||x_1 - x_*||^2 <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2 ")
        println("\tTheoretical guarantee:\t ||x_1 - x_*||^2 <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_polyak_steps_in_distance_to_optimum(1.0, 0.1, 2 / (1.0 + 0.1); solver=Clarabel.Optimizer, verbose=true)
