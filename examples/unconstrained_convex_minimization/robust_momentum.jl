using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_robust_momentum(mu, L, lam; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_robust_momentum`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly-convex.

# Performance metric

This code computes a worst-case guarantee for the **robust momentum method** (RMM).
That is, it computes the smallest possible $\tau(n, \mu, L, \lambda)$ such that the guarantee

```math
v(x_{n+1}) \leqslant \tau(n, \mu, L, \lambda) v(x_{n}),
```

is valid, where $x_n$ is the $n^{\mathrm{th}}$ iterate of the RMM, and $x_\star$ is a minimizer
of $f$. The function $v(.)$ is a well-chosen Lyapunov defined as follows,

```math
    \begin{aligned}
        v(x_t) & = & l\|z_t - x_\star\|^2 + q_t, \\
        q_t & = & (L - \mu) \left(f(x_t) - f_\star - \frac{\mu}{2}\|y_t - x_\star\|^2 - \frac{1}{2}\|\nabla f(y_t) - \mu (y_t - x_\star)\|^2 \right),
    \end{aligned}
```
with $\kappa = \frac{\mu}{L}$, $\rho = \lambda (1 - \frac{1}{\kappa}) + (1 - \lambda) \left(1 - \frac{1}{\sqrt{\kappa}}\right)$, and $l = \mu^2  \frac{\kappa - \kappa \rho^2 - 1}{2 \rho (1 - \rho)}$.

# Algorithm


For $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        x_{t+1} & = & x_{t} + \beta (x_t - x_{t-1}) - \alpha \nabla f(y_t), \\
        y_{t+1} & = & y_{t} + \gamma (x_t - x_{t-1}),
    \end{aligned}
```
with $x_{-1}, x_0 \in \mathrm{R}^d$,
and with parameters $\alpha = \frac{\kappa (1 - \rho^2)(1 + \rho)}{L}$, $\beta = \frac{\kappa \rho^3}{\kappa - 1}$, $\gamma = \frac{\rho^2}{(\kappa - 1)(1 - \rho)^2(1 + \rho)}$.

# Theoretical guarantee


A convergence guarantee (empirically tight) is obtained in [1, Theorem 1],

```math
v(x_{n+1}) \leqslant \rho^2 v(x_n),
```

with $\rho = \lambda (1 - \frac{1}{\kappa}) + (1 - \lambda) \left(1 - \frac{1}{\sqrt{\kappa}}\right)$.

# References


[[1] S. Cyrus, B. Hu, B. Van Scoy, L. Lessard (2018).
A robust accelerated optimization algorithm for strongly convex functions.
American Control Conference (ACC).](https://arxiv.org/pdf/1710.04753.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `lam`: if $\lambda=1$ it is the gradient descent, if $\lambda=0$, it is the Triple Momentum Method.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_robust_momentum(0.1, 1.0, 0.2; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.528555, 0.528555)
```
"""
function wc_robust_momentum(mu, L, lam; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    x1 = set_initial_point!(problem)


    kappa = L / mu
    rho = lam * (1 - 1 / kappa) + (1 - lam) * (1 - 1 / sqrt(kappa))
    alpha = kappa * (1 - rho)^2 * (1 + rho) / L
    beta = kappa * rho^3 / (kappa - 1)
    gamma = rho^3 / ((kappa - 1) * (1 - rho)^2 * (1 + rho))
    l = mu^2 * (kappa - kappa * rho^2 - 1) / (2 * rho * (1 - rho))


    y0 = x1 + gamma * (x1 - x0)
    g0, f0 = oracle!(func, y0)
    x2 = x1 + beta * (x1 - x0) - alpha * g0
    y1 = x2 + gamma * (x2 - x1)
    g1, f1 = oracle!(func, y1)
    x3 = x2 + beta * (x2 - x1) - alpha * g1

    z1 = (x2 - (rho^2) * x1) / (1 - rho^2)
    z2 = (x3 - (rho^2) * x2) / (1 - rho^2)


    q0 = (L - mu) * (f0 - fs - mu / 2 * (y0 - xs)^2) - 1 / 2 * (g0 - mu * (y0 - xs))^2
    q1 = (L - mu) * (f1 - fs - mu / 2 * (y1 - xs)^2) - 1 / 2 * (g1 - mu * (y1 - xs))^2
    initLyapunov = l * (z1 - xs)^2 + q0
    finalLyapunov = l * (z2 - xs)^2 + q1


    set_initial_condition!(problem, initLyapunov <= 1)


    set_performance_metric!(problem, finalLyapunov)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = rho^2


    if verbose
        println("*** Example file: worst-case performance of the Robust Momentum Method ***")
        println("\tPEPit guarantee:\t v(x_(n+1)) <= $(round(pepit_tau, digits=6)) v(x_n)")
        println("\tTheoretical guarantee:\t v(x_(n+1)) <= $(round(theoretical_tau, digits=6)) v(x_n)")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_robust_momentum(0.1, 1.0, 0.2; solver=Clarabel.Optimizer, verbose=true)
