using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_douglas_rachford_splitting(L, mu, alpha, theta; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_douglas_rachford_splitting`.

Consider the monotone inclusion problem

```math
\mathrm{Find}\, x:\, 0\in Ax + Bx,
```

where $A$ is $L$-Lipschitz and maximally monotone and $B$ is (maximally) $\mu$-strongly
monotone. We denote by $J_{\alpha A}$ and $J_{\alpha B}$ the resolvents of respectively
$\alpha A$ and $\alpha B$.

# Performance metric

This code computes a worst-case guarantee for the **Douglas-Rachford splitting** (DRS).
That is, given two initial points $w^{(0)}_t$ and $w^{(1)}_t$,
this code computes the smallest possible $\tau(L, \mu, \alpha, \theta)$
(a.k.a. "contraction factor") such that the guarantee

```math
\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\|^2 \leqslant \tau(L, \mu, \alpha, \theta) \|w^{(0)}_{t} - w^{(1)}_{t}\|^2,
```

is valid, where $w^{(0)}_{t+1}$ and $w^{(1)}_{t+1}$ are obtained after one iteration of DRS from
respectively $w^{(0)}_{t}$ and $w^{(1)}_{t}$.

In short, for given values of $L$, $\mu$, $\alpha$ and $\theta$, the contraction
factor $\tau(L, \mu, \alpha, \theta)$ is computed as the worst-case value of
$\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\|^2$ when $\|w^{(0)}_{t} - w^{(1)}_{t}\|^2 \leqslant 1$.

# Algorithm
One iteration of the Douglas-Rachford splitting is described as follows,
for $t \in \{ 0, \dots, n-1\}$,

```math
    \begin{aligned}
        x_{t+1} & = & J_{\alpha B} (w_t),\\
        y_{t+1} & = & J_{\alpha A} (2x_{t+1}-w_t),\\
        w_{t+1} & = & w_t - \theta (x_{t+1}-y_{t+1}).
    \end{aligned}
```
# Theoretical guarantee
Theoretical worst-case guarantees can be found in [1, section 4, Theorem 4.3].
Since the results of [2] tighten that of [1], we compare with [2, Theorem 4.3] below. The theoretical results
are complicated and we do not copy them here.

# References
The detailed PEP methodology for studying operator splitting is provided in [2].

[[1] W. Moursi, L. Vandenberghe (2019). Douglas-Rachford Splitting for the Sum of a Lipschitz Continuous and
a Strongly Monotone Operator. Journal of Optimization Theory and Applications 183, 179-198.](https://arxiv.org/pdf/1805.09396.pdf)

[[2] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020). Operator splitting performance estimation:
Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3), 2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `alpha`: algorithm parameter used in the update rule.
- `theta`: relaxation or averaging parameter used in the update rule.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
wc_douglas_rachford_splitting(1.0, 0.1, 1.3, 0.9; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.928771, 0.928771)
```
"""
function wc_douglas_rachford_splitting(L, mu, alpha, theta; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, LipschitzStronglyMonotoneOperatorCheap, OrderedDict("L" => L, "mu" => 0))
    B = declare_function!(problem, StronglyMonotoneOperator, OrderedDict("mu" => mu))


    w0 = set_initial_point!(problem)
    w1 = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w1)^2 <= 1)


    x0, _, _ = proximal_step!(w0, B, alpha)
    y0, _, _ = proximal_step!(2 * x0 - w0, A, alpha)
    z0 = w0 - theta * (x0 - y0)


    x1, _, _ = proximal_step!(w1, B, alpha)
    y1, _, _ = proximal_step!(2 * x1 - w1, A, alpha)
    z1 = w1 - theta * (x1 - y1)


    set_performance_metric!(problem, (z0 - z1)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    mu = alpha * mu
    L = alpha * L
    c = sqrt(((2 * (theta - 1) * mu + theta - 2)^2 + L^2 * (theta - 2 * (mu + 1))^2) / (L^2 + 1))
    if theta * (theta + c) / (mu + 1)^2 / c * (
            c + mu * ((2 * (theta - 1) * mu + theta - 2) - L^2 * (theta - 2 * (mu + 1))) / (L^2 + 1)) >= 0
        theoretical_tau = ((theta + c) / 2 / (mu + 1))^2
    elseif (L <= 1) && (mu >= (L^2 + 1) / (L - 1)^2) && (theta <= -(2 * (mu + 1) * (L + 1) *
                                                                    (mu + (mu - 1) * L^2 - 2 * mu * L - 1)) / (
                                                             mu + L * (L^2 + L + 1) + 2 * mu^2 * (L - 1)
                                                             + mu * L * (1 - (L - 3) * L) + 1))
        theoretical_tau = (1 - theta * (L + mu) / (L + 1) / (mu + 1))^2
    else
        theoretical_tau = (2 - theta) / 4 / mu / (L^2 + 1) * (
            theta * (1 - 2 * mu + L^2) - 2 * mu * (L^2 - 1)) *
                          (theta * (1 + 2 * mu + L^2) - 2 * (mu + 1) * (L^2 + 1)) / (
            theta * (1 + 2 * mu - L^2) - 2 * (mu + 1) * (1 - L^2))
    end


    if verbose
        println("*** Example file: worst-case performance of the Douglas Rachford Splitting***")
        println("\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(pepit_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
        println("\tTheoretical guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(theoretical_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_douglas_rachford_splitting(1.0, 0.1, 1.3, 0.9; verbose=true)
