using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_three_operator_splitting(L, mu, beta, alpha, theta; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_three_operator_splitting`.

Consider the monotone inclusion problem

```math
\mathrm{Find}\, x:\, 0\in Ax + Bx + Cx,
```

where $A$ is maximally monotone, $B$ is $\beta$-cocoercive and C is the gradient of some
$L$-smooth $\mu$-strongly convex function. We denote by $J_{\alpha A}$ and $J_{\alpha B}$
the resolvents of respectively $\alpha A$ and $\alpha B$.

# Performance metric

This code computes a worst-case guarantee for the **three operator splitting** (TOS).
That is, given two initial points $w^{(0)}_t$ and $w^{(1)}_t$,
this code computes the smallest possible $\tau(L, \mu, \beta, \alpha, \theta)$
(a.k.a. "contraction factor") such that the guarantee

```math
\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\|^2 \leqslant \tau(L, \mu, \beta, \alpha, \theta) \|w^{(0)}_{t} - w^{(1)}_{t}\|^2,
```

is valid, where $w^{(0)}_{t+1}$ and $w^{(1)}_{t+1}$ are obtained after one iteration of TOS from
respectively $w^{(0)}_{t}$ and $w^{(1)}_{t}$.

In short, for given values of $L$, $\mu$, $\beta$, $\alpha$ and $\theta$,
the contraction factor $\tau(L, \mu, \beta, \alpha, \theta)$ is computed as the worst-case value of
$\|w^{(0)}_{t+1} - w^{(1)}_{t+1}\|^2$ when $\|w^{(0)}_{t} - w^{(1)}_{t}\|^2 \leqslant 1$.

# Algorithm

One iteration of the algorithm is described in [1]. For $t \in \{ 0, \dots, n-1\}$,

```math
    \begin{aligned}
        x_{t+1} & = & J_{\alpha B} (w_t),\\
        y_{t+1} & = & J_{\alpha A} (2x_{t+1} - w_t - C x_{t+1}),\\
        w_{t+1} & = & w_t - \theta (x_{t+1} - y_{t+1}).
    \end{aligned}
```
# References
The TOS was proposed in [1],
the analysis of such operator splitting methods using PEPs was proposed in [2].

[[1] D. Davis, W. Yin (2017). A three-operator splitting scheme and its optimization applications.
Set-valued and variational analysis, 25(4), 829-858.](https://arxiv.org/pdf/1504.01032.pdf)

[[2] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020). Operator splitting performance estimation:
Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3), 2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `beta`: operator or algorithm parameter used in the model.
- `alpha`: algorithm parameter used in the update rule.
- `theta`: relaxation or averaging parameter used in the update rule.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: no theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_three_operator_splitting(1.0, 0.1, 1.0, 0.9, 1.3; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.779689, nothing)
```
"""
function wc_three_operator_splitting(L, mu, beta, alpha, theta; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, MonotoneOperator, OrderedDict())
    B = declare_function!(problem, CocoerciveOperator, OrderedDict("beta" => beta))
    C = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


    w0 = set_initial_point!(problem)
    w1 = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w1)^2 <= 1)


    x0, _, _ = proximal_step!(w0, B, alpha)
    y0, _, _ = proximal_step!(2 * x0 - w0 - alpha * gradient!(C, x0), A, alpha)
    z0 = w0 - theta * (x0 - y0)


    x1, _, _ = proximal_step!(w1, B, alpha)
    y1, _, _ = proximal_step!(2 * x1 - w1 - alpha * gradient!(C, x1), A, alpha)
    z1 = w1 - theta * (x1 - y1)


    set_performance_metric!(problem, (z0 - z1)^2)


    pepit_verbose = verbose != false
    pepit_tau = solve!(problem; solver=solver, verbose=pepit_verbose)


    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case contraction factor of the Three Operator Splitting ***")
        println("\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(pepit_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_three_operator_splitting(1.0, 0.1, 1.0, 0.9, 1.3; verbose=true)
