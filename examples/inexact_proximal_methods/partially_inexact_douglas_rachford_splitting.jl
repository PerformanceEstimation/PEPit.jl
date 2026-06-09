using PEPit, OrderedCollections, Clarabel


@doc raw"""
    wc_partially_inexact_douglas_rachford_splitting(mu, L, n, gamma, sigma; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_partially_inexact_douglas_rachford_splitting`.

Consider the composite strongly convex minimization problem,

```math
F_\star \triangleq \min_x \left\{ F(x) \equiv f(x) + g(x) \right\}
```

where $f$ is $L$-smooth and $\mu$-strongly convex, and $g$ is closed convex and proper. We
denote by $x_\star = \arg\min_x F(x)$ the minimizer of $F$.
The (exact) proximal operator of $g$, and an approximate version of the proximal operator of
$f$ are assumed to be available.

# Performance metric

This code computes a worst-case guarantee for a **partially inexact Douglas-Rachford Splitting** (DRS). That is, it
computes the smallest possible $\tau(n,L,\mu,\sigma,\gamma)$ such that the guarantee

```math
\|z_{n} - z_\star\|^2 \leqslant \tau(n,L,\mu,\sigma,\gamma)  \|z_0 - z_\star\|^2
```

is valid, where $z_n$ is the output of the DRS (initiated at $x_0$),
$z_\star$ is its fixed point,
$\gamma$ is a step-size,
and $\sigma$ is the level of inaccuracy.

# Algorithm
The partially inexact Douglas-Rachford splitting under consideration is described by

```math
    \begin{aligned}
         x_{t} && \approx_{\sigma} \arg\min_x \left\{ \gamma f(x)+\frac{1}{2} \|x-z_t\|^2 \right\},\\
         y_{t} && = \arg\min_y \left\{ \gamma g(y)+\frac{1}{2} \|y-(x_t-\gamma \nabla f(x_t))\|^2 \right\},\\
         z_{t+1} && = z_t + y_t - x_t.
    \end{aligned}
```
More precisely, the notation "$\approx_{\sigma}$" correspond to require the existence of some
$e_{t}$ such that

```math
    \begin{aligned}
         x_{t} && = z_t - \gamma (\nabla f(x_t) - e_t),\\
         y_{t} && =  \arg\min_y \left\{ \gamma g(y)+\frac{1}{2} \|y-(x_t-\gamma \nabla f(x_t))\|^2 \right\},\\
          && \text{with } \|e_t\|^2 \leqslant \frac{\sigma^2}{\gamma^2}\|y_{t} - z_t + \gamma \nabla f(x_t) \|^2,\\
         z_{t+1} && = z_t + y_t - x_t.
    \end{aligned}
```
# Theoretical guarantee
The following **tight** theoretical bound is due to [2, Theorem 5.1]:

```math
\|z_{n} - z_\star\|^2  \leqslant \max\left(\frac{1 - \sigma + \gamma \mu \sigma}{1 - \sigma + \gamma \mu},
```
                         \frac{\sigma + (1 - \sigma) \gamma L}{1 + (1 - \sigma) \gamma L)}\right)^{2n} \|z_0 - z_\star\|^2.

# References
The method is from [1], its PEP formulation and the worst-case analysis from [2],
see [2, Section 4.4] for more details.

[[1] J. Eckstein and W. Yao (2018).
Relative-error approximate versions of Douglas-Rachford splitting and special cases of the ADMM.
Mathematical Programming, 170(2), 417-444.](https://link.springer.com/article/10.1007/s10107-017-1160-5)

[[2] M. Barre, A. Taylor, F. Bach (2020).
Principled analyses and design of first-order methods with inexact proximal operators,
arXiv 2006.06041v2.](https://arxiv.org/pdf/2006.06041v2.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `gamma`: step-size parameter.
- `sigma`: noise parameter.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_partially_inexact_douglas_rachford_splitting(0.1, 5, 5, 1.4, 0.2; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.281206, 0.281206)
```
"""
function wc_partially_inexact_douglas_rachford_splitting(mu, L, n, gamma, sigma; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    g = declare_function!(problem, ConvexFunction, OrderedDict())


    F = f + g


    xs = stationary_point!(F)


    zs = xs + gamma * gradient!(f, xs)


    z0 = set_initial_point!(problem)


    set_initial_condition!(problem, (z0 - zs)^2 <= 1)


    z = z0
    for _ in 1:n

        x, dfx, _, _, _, _, eps_var = inexact_proximal_step!(z, f, gamma; opt="PD_gapII")


        y, _, _ = proximal_step!(x - gamma * dfx, g, gamma)


        add_constraint!(f, eps_var <= 1 / 2 * (sigma * (y - z + gamma * dfx))^2)


        z = z + (y - x)
    end


    set_performance_metric!(problem, (z - zs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max(((1 - sigma + gamma * mu * sigma) / (1 - sigma + gamma * mu))^2,
                          ((sigma + (1 - sigma) * gamma * L) / (1 + (1 - sigma) * gamma * L))^2)^n


    if verbose

        println("*** Example file: worst-case performance of the partially inexact Douglas Rachford splitting ***")
        println("\tPEPit guarantee:\t ||z_n - z_*||^2 <= $(round(pepit_tau, digits=6)) ||z_0 - z_*||^2")
        println("\tTheoretical guarantee:\t ||z_n - z_*||^2 <= $(round(theoretical_tau, digits=6)) ||z_0 - z_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_partially_inexact_douglas_rachford_splitting(0.1, 5, 5, 1.4, 0.2; verbose=true)
