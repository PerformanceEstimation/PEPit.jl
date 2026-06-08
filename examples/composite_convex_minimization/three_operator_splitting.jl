using PEPit, OrderedCollections


@doc raw"""
    wc_three_operator_splitting(mu1, L1, L3, alpha, theta, n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_three_operator_splitting`.

Consider the composite convex minimization problem,

```math
F_\star \triangleq \min_x \{F(x) \equiv f_1(x) + f_2(x) + f_3(x)\}
```

where, $f_1$ is $L_1$-smooth and $\mu_1$-strongly convex,
$f_2$ is closed, convex and proper,
and $f_3$ is $L_3$-smooth convex.
Proximal operators are assumed to be available for $f_1$ and $f_2$.

# Performance metric

This code computes the worst-case guarantee of the contraction achieved by the **Three Operator Splitting (TOS)**.
That is, it computes the smallest possible $\tau(n, L_1, L_3, \mu_1)$ such that the guarantee

```math
\|w^{(0)}_{n} - w^{(1)}_{n}\|^2 \leqslant \tau(n, L_1, L_3, \mu_1, \alpha, \theta) \|w^{(0)}_{0} - w^{(1)}_{0}\|^2
```

is valid, where $w^{(0)}_{0}$ and $w^{(1)}_{0}$ are two different starting points
and $w^{(0)}_{n}$ and $w^{(1)}_{n}$ are the two corresponding $n^{\mathrm{th}}$ outputs of TOS.

In short, for given values of $n$, $L_1$, $L_3$, $\mu_1$, $\alpha$
and $\theta$, the contraction factor $\tau(n, L_1, L_3, \mu_1, \alpha, \theta)$
is computed as the worst-case value of $\|w^{(0)}_{n} - w^{(1)}_{n}\|^2$
when $\|w^{(0)}_{0} - w^{(1)}_{0}\|^2 \leqslant 1$.

# Algorithm

One iteration of the algorithm is described in [1]. For $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        x_t & = & \mathrm{prox}_{\alpha, f_2}(w_t), \\
        y_t & = & \mathrm{prox}_{\alpha, f_1}(2 x_t - w_t - \alpha \nabla f_3(x_t)), \\
        w_{t+1} & = & w_t + \theta (y_t - x_t).
    \end{aligned}
```
# References
The TOS was introduced in [1].

[[1] D. Davis, W. Yin (2017).
A three-operator splitting scheme and its optimization applications.
Set-valued and variational analysis, 25(4), 829-858.](https://arxiv.org/pdf/1504.01032.pdf)

# Arguments
- `mu1`: the strong convexity parameter of function $f_1$.
- `L1`: the smoothness parameter of function $f_1$.
- `L3`: the smoothness parameter of function $f_3$.
- `alpha`: algorithm parameter used in the update rule.
- `theta`: relaxation or averaging parameter used in the update rule.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: no theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_three_operator_splitting(0.1, 10.0, L3, alpha, 1.0, 4; verbose=true)
```
"""
function wc_three_operator_splitting(mu1, L1, L3, alpha, theta, n; verbose=true)


    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu1, "L" => L1); reuse_gradient=true)
    f2 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)
    f3 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L3); reuse_gradient=true)


    w0 = set_initial_point!(problem)
    w0p = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w0p)^2 <= 1)


    w = w0
    for _ in 1:n

        x, _, _ = proximal_step!(w, f2, alpha)


        gx, _ = oracle!(f3, x)


        y, _, _ = proximal_step!(2 * x - w - alpha * gx, f1, alpha)


        w = w + theta * (y - x)
    end


    wp = w0p
    for _ in 1:n

        xp, _, _ = proximal_step!(wp, f2, alpha)


        gxp, _ = oracle!(f3, xp)


        yp, _, _ = proximal_step!(2 * xp - wp - alpha * gxp, f1, alpha)


        wp = wp + theta * (yp - xp)
    end


    set_performance_metric!(problem, (w - wp)^2)


    pepit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = nothing


    if verbose

        println("*** Example file: worst-case performance of the Three Operator Splitting in distance ***")
        println("\tPEPit guarantee:\t ||w^1_n - w^0_n||^2 <= $(round(pepit_tau, digits=6)) ||w^1_0 - w^0_0||^2")
    end


    return pepit_tau, theoretical_tau
end


L3 = 1.0


alpha = 1 / L3


pepit_tau, theoretical_tau = wc_three_operator_splitting(0.1, 10.0, L3, alpha, 1.0, 4; verbose=true)
