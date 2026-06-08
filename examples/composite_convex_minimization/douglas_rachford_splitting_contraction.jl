using PEPit
using OrderedCollections

@doc raw"""
    wc_douglas_rachford_splitting_contraction(mu, L, alpha, theta, n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_douglas_rachford_splitting_contraction`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f_1(x) + f_2(x) \}
```

where $f_1(x)$ is $L$-smooth and $\mu$-strongly convex, and $f_2$ is convex,
closed and proper. Both proximal operators are assumed to be available.

# Performance metric

This code computes a worst-case guarantee for the **Douglas Rachford Splitting (DRS)** method.
That is, it computes the smallest possible $\tau(\mu,L,\alpha,\theta,n)$ such that the guarantee

```math
\|w_1 - w_1'\|^2 \leqslant \tau(\mu,L,\alpha,\theta,n) \|w_0 - w_0'\|^2.
```

is valid, where $x_n$ is the output of the **Douglas Rachford Splitting method**. It is a contraction
factor computed when the algorithm is started from two different points $w_0$ and $w_0$.

# Algorithm


Our notations for the DRS method are as follows [3, Section 7.3], for $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        x_t & = & \mathrm{prox}_{\alpha f_2}(w_t), \\
        y_t & = & \mathrm{prox}_{\alpha f_1}(2x_t - w_t), \\
        w_{t+1} & = & w_t + \theta (y_t - x_t).
    \end{aligned}
```
# Theoretical guarantee


The **tight** theoretial guarantee is obtained in [2, Theorem 2]:

```math
\|w_1 - w_1'\|^2 \leqslant  \max\left(\frac{1}{1 + \mu \alpha}, \frac{\alpha L }{1 + L \alpha}\right)^{2n} \|w_0 - w_0'\|^2
```

for when $\theta=1$.

# References


Details on the SDP formulations can be found in

[[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and optimal parameter selection.
SIAM Journal on Optimization, 30(3), 2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

When $\theta = 1$, the bound can be compared with that of [2, Theorem 2]

[[2] P. Giselsson, and S. Boyd (2016).
Linear convergence and metric selection in Douglas-Rachford splitting and ADMM.
IEEE Transactions on Automatic Control, 62(2), 532-544.](https://arxiv.org/pdf/1410.8479.pdf)

A description for the DRS method can be found in [3, 7.3]

[[3] E. Ryu, S. Boyd (2016).
A primer on monotone operator methods.
Applied and Computational Mathematics 15(1), 3-43.](https://web.stanford.edu/~boyd/papers/pdf/monotone_primer.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `alpha`: algorithm parameter used in the update rule.
- `theta`: relaxation or averaging parameter used in the update rule.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
PEPit_tau, theoretical_tau = wc_douglas_rachford_splitting_contraction(0.1, 1.0, 3.0, 1.0, 2; verbose=true)
```
"""
function wc_douglas_rachford_splitting_contraction(mu, L, alpha, theta, n; verbose=true)
    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L); reuse_gradient=true)
    f2 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)


    w0 = set_initial_point!(problem)
    w0p = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w0p)^2 <= 1)


    w = w0
    for _ in 1:n
        x, _, _ = proximal_step!(w, f2, alpha)
        y, _, _ = proximal_step!(2 * x - w, f1, alpha)
        w = w + theta * (y - x)
    end


    wp = w0p
    for _ in 1:n
        xp, _, _ = proximal_step!(wp, f2, alpha)
        yp, _, _ = proximal_step!(2 * xp - wp, f1, alpha)
        wp = wp + theta * (yp - xp)
    end


    set_performance_metric!(problem, (w - wp)^2)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = nothing
    if theta == 1
        theoretical_tau = (max(1 / (1 + mu * alpha), (alpha * L) / (1 + alpha * L)))^(2n)
    end

    if verbose
        println("*** Example file: worst-case performance of the Douglas-Rachford splitting in distance ***")
        println("\tPEPit guarantee:\t ||w - wp||^2 <= $(round(PEPit_tau, digits=6)) ||w0 - w0p||^2")
        if theta == 1
            println("\tTheoretical guarantee:\t ||w - wp||^2 <= $(round(theoretical_tau, digits=6)) ||w0 - w0p||^2")
        end
    end

    return PEPit_tau, theoretical_tau
end


PEPit_tau, theoretical_tau = wc_douglas_rachford_splitting_contraction(0.1, 1.0, 3.0, 1.0, 2; verbose=true)
