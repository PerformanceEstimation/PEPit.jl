using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_douglas_rachford_splitting_2(beta, mu, alpha, theta; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_douglas_rachford_splitting_2`.

Consider the monotone inclusion problem

```math
\mathrm{Find}\, x:\, 0\in Ax + Bx,
```

where $A$ is $\beta$-cocoercive and maximally monotone
and $B$ is (maximally) $\mu$-strongly monotone.
We denote by $J_{\alpha A}$ and $J_{\alpha B}$ the resolvents of respectively
$\alpha A$ and $\alpha B$.

# Performance metric

This code computes a worst-case guarantee for the **Douglas-Rachford splitting** (DRS).
That is, given two initial points $w^{(0)}_t$ and $w^{(1)}_t$,
this code computes the smallest possible $\tau(\beta, \mu, \alpha, \theta)$
(a.k.a. "contraction factor") such that the guarantee

```math
\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\|^2 \leqslant \tau(\beta, \mu, \alpha, \theta) \|w^{(0)}_{t} - w^{(1)}_{t}\|^2,
```

is valid, where $w^{(0)}_{t+1}$ and $w^{(1)}_{t+1}$ are obtained after one iteration of DRS from
respectively $w^{(0)}_{t}$ and $w^{(1)}_{t}$.

In short, for given values of $\beta$, $\mu$, $\alpha$ and $\theta$, the contraction
factor $\tau(\beta, \mu, \alpha, \theta)$ is computed as the worst-case value of
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
Theoretical worst-case guarantees can be found in [1, section 4, Theorem 4.1].

# References
The detailed PEP methodology for studying operator splitting is provided in [1].

[[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020). Operator splitting performance estimation:
Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3), 2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

# Arguments
- `beta`: operator or algorithm parameter used in the model.
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
wc_douglas_rachford_splitting_2(1.2, 0.1, 0.3, 1.5; verbose=true)
```
"""
function wc_douglas_rachford_splitting_2(beta, mu, alpha, theta; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, CocoerciveOperator, OrderedDict("beta" => beta))
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
    beta = alpha * beta
    if mu * beta - mu + beta < 0 && theta <= 2 * (beta + 1) * (mu - beta - mu * beta) /
                                            (mu + mu * beta - beta - beta^2 - 2 * mu * beta^2)
        theoretical_tau = (1 - theta * beta / (beta + 1))^2
    elseif mu * beta - mu - beta > 0 && theta <= 2 * (mu^2 + beta^2 + mu * beta + mu + beta - mu^2 * beta^2) /
                                                 (mu^2 + beta^2 + mu^2 * beta + mu * beta^2 + mu + beta - 2 * mu^2 * beta^2)
        theoretical_tau = (1 - theta * (1 + mu * beta) / (mu + 1) / (beta + 1))^2
    elseif theta >= 2 * (mu * beta + mu + beta) / (2 * mu * beta + mu + beta)
        theoretical_tau = (1 - theta)^2
    elseif mu * beta + mu - beta < 0 && theta <= 2 * (mu + 1) * (beta - mu - mu * beta) /
                                                 (beta + mu * beta - mu - mu^2 - 2 * mu^2 * beta)
        theoretical_tau = (1 - theta * mu / (mu + 1))^2
    else
        theoretical_tau = (2 - theta) / 4 / mu * ((2 - theta) * mu * (beta + 1) + theta * beta * (1 - mu)) *
                          ((2 - theta) * beta * (mu + 1) + theta * mu * (1 - beta)) / mu / beta /
                          (2 * mu * beta * (1 - theta) + (2 - theta) * (mu + beta + 1))
    end


    if verbose != -1
        println("*** Example file: worst-case performance of the Douglas Rachford Splitting***")
        println("\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(pepit_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
        println("\tTheoretical guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(theoretical_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_douglas_rachford_splitting_2(1.2, 0.1, 0.3, 1.5; verbose=true)
