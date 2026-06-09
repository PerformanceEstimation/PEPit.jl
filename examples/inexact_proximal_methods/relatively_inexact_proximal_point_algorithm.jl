using PEPit, OrderedCollections, Clarabel, OffsetArrays


@doc raw"""
    wc_relatively_inexact_proximal_point_algorithm(n, gamma, sigma; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_relatively_inexact_proximal_point_algorithm`.

Consider the (possibly non-smooth) convex minimization problem,

```math
f_\star \triangleq \min_x f(x)
```

where $f$ is closed, convex, and proper. We denote by $x_\star$ some optimal point of $f$ (hence
$0\in\partial f(x_\star)$). We further assume that one has access to an inexact version of the proximal
operator of $f$, whose level of accuracy is controlled by some parameter $\sigma\geqslant 0$.

# Performance metric

This code computes a worst-case guarantee for an **inexact proximal point method**. That is, it computes the
smallest possible $\tau(n, \gamma, \sigma)$ such that the guarantee

```math
f(x_n) - f(x_\star) \leqslant \tau(n, \gamma, \sigma) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the method, $\gamma$ is some step-size, and $\sigma$ is
the level of accuracy of the approximate proximal point oracle.

# Algorithm
The approximate proximal point method under consideration is described by

```math
x_{t+1} \approx_{\sigma} \arg\min_x \left\{ \gamma f(x)+\frac{1}{2} \|x-x_t\|^2 \right\},
```

where the notation "$\approx_{\sigma}$" corresponds to require the existence of some vector
$s_{t+1}\in\partial f(x_{t+1})$ and $e_{t+1}$ such that

```math
x_{t+1}  =  x_t - \gamma s_{t+1} + e_{t+1} \quad \quad \text{with }\|e_{t+1}\|^2  \leqslant  \sigma^2\|x_{t+1} - x_t\|^2.
```

We note that the case $\sigma=0$ implies $e_{t+1}=0$ and this operation reduces to a standard proximal
step with step-size $\gamma$.

# Theoretical guarantee
The following (empirical) upper bound is provided in [1, Section 3.5.1],

```math
f(x_n) - f(x_\star) \leqslant \frac{1 + \sigma}{4 \gamma n^{\sqrt{1 - \sigma^2}}}\|x_0 - x_\star\|^2.
```

# References
The precise formulation is presented in [1, Section 3.5.1].

[[1] M. Barre, A. Taylor, F. Bach (2020).
Principled analyses and design of first-order methods with inexact proximal operators.
arXiv 2006.06041v2.](https://arxiv.org/pdf/2006.06041.pdf)

# Arguments
- `n`: number of iterations.
- `gamma`: step-size parameter.
- `sigma`: accuracy parameter of the proximal point computation.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_relatively_inexact_proximal_point_algorithm(8, 10, 0.65; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.007678, 0.008494)
```
"""
function wc_relatively_inexact_proximal_point_algorithm(n, gamma, sigma; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(f)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = OffsetVector([x0 for _ in 0:n], 0:n)


    for i in 0:n-1

        x[i + 1], _, fx, _, _, _, epsVar = inexact_proximal_step!(x[i], f, gamma; opt="PD_gapII")


        add_constraint!(f, epsVar <= (sigma * (x[i + 1] - x[i]))^2 / 2)
    end


    set_performance_metric!(problem, value!(f, x[n]) - value!(f, xs))


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 + sigma) / (4 * gamma * n^sqrt(1 - sigma^2))


    if verbose

        println("*** Example file: worst-case performance of an inexact proximal point method in distance in function values ***")
        println("\tPEPit guarantee:\t f(x_n) - f(x_*) <= $(round(pepit_tau, digits=7)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n) - f(x_*) <= $(round(theoretical_tau, digits=7)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_relatively_inexact_proximal_point_algorithm(8, 10, 0.65; verbose=true)
