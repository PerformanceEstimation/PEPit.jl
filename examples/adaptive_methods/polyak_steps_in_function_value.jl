using PEPit, Clarabel
using OrderedCollections

@doc raw"""
    wc_polyak_steps_in_function_value(L, mu, gamma; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_polyak_steps_in_function_value`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex, and $x_\star=\arg\min_x f(x)$.

# Performance metric

This code computes a worst-case guarantee for a variant of a **gradient method** relying on **Polyak step-sizes**.
That is, it computes the smallest possible $\tau(L, \mu, \gamma)$ such that the guarantee

```math
f(x_{t+1}) - f_\star \leqslant \tau(L, \mu, \gamma) (f(x_t) - f_\star)
```

is valid, where $x_t$ is the output of the gradient method with PS and $\gamma$ is the effective value
of the step-size of the gradient method.

In short, for given values of $L$, $\mu$, and $\gamma$, $\tau(L, \mu, \gamma)$ is
computed as the worst-case value of $f(x_{t+1})-f_\star$ when $f(x_t)-f_\star \leqslant 1$.

# Algorithm

Gradient descent is described by

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size. The Polyak step-size rule under consideration here corresponds to choosing
of $\gamma$ satisfying:

```math
\|\nabla f(x_t)\|^2 = 2  L (2 - L \gamma) (f(x_t) - f_\star).
```

# Theoretical guarantee

The gradient method with the variant of Polyak step-sizes under consideration enjoys the
**tight** theoretical guarantee [1, Proposition 2]:

```math
f(x_{t+1})-f_\star \leqslant  \tau(L,\mu,\gamma) (f(x_{t})-f_\star),
```

where $\gamma$ is the effective step-size used at iteration $t$ and

```math
    \begin{aligned}
        \tau(L,\mu,\gamma) & = & \left\{\begin{array}{ll} (\gamma L - 1)  (L \gamma  (3 - \gamma (L + \mu)) - 1)  & \text{if } \gamma\in[\tfrac{1}{L},\tfrac{2L-\mu}{L^2}],\\
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
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_polyak_steps_in_function_value(1.0, 0.1, 2 / (1 + 0.1); verbose=true)
```
"""
function wc_polyak_steps_in_function_value(L, mu, gamma; verbose=true)
    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu); reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    g0, f0 = oracle!(func, x0)


    set_initial_condition!(problem, f0 - fs <= 1)
    add_constraint!(problem, g0^2 == 2 * L * (2 - L * gamma) * (f0 - fs))


    x1 = x0 - gamma * g0
    g1, f1 = oracle!(func, x1)


    set_performance_metric!(problem, f1 - fs)

    pepit_tau = solve!(problem; solver = Clarabel.Optimizer, verbose=verbose)

    theoretical_tau = (1 / L <= gamma <= (2 * L - mu) / L^2) ?
        (gamma * L - 1) * (L * gamma * (3 - gamma * (L + mu)) - 1) :
        0.0

    if verbose
        println("*** Example file: worst-case performance of Polyak steps ***")
        println("\tPEPit guarantee:\t f(x_1) - f_* <= $(round(pepit_tau, digits=6)) (f(x_0) - f_*) ")
        println("\tTheoretical guarantee:\t f(x_1) - f_* <= $(round(theoretical_tau, digits=6)) (f(x_0) - f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_polyak_steps_in_function_value(1.0, 0.1, 2 / (1 + 0.1); verbose=true)
