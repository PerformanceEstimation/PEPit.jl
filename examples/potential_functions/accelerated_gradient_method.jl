using PEPit
using OrderedCollections

@doc raw"""
    wc_accelerated_gradient_method(L, gamma, lam; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_gradient_method`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code verifies a worst-case guarantee for an **accelerated gradient method**. That is, it verifies
that the Lyapunov (or potential/energy) function

```math
V_n \triangleq \lambda_n^2 (f(x_n) - f_\star) + \frac{L}{2} \|z_n - x_\star\|^2
```

is decreasing along all trajectories and all smooth convex function $f$ (i.e., in the worst-case):

```math
V_{n+1} \leqslant V_n,
```

where $x_{n+1}$, $z_{n+1}$, and $\lambda_{n+1}$ are obtained from one iteration of
the accelerated gradient method below, from some arbitrary $x_{n}$, $z_{n}$, and $\lambda_{n}$.

# Algorithm
One iteration of accelerated gradient method is described by

```math
\begin{aligned}
    \text{Set: }\lambda_{n+1} & = & \frac{1}{2} \left(1 + \sqrt{4\lambda_n^2 + 1}\right), \tau_n & = & \frac{1}{\lambda_{n+1}},
    \text{ and } \eta_n & = & \frac{\lambda_{n+1}^2 - \lambda_{n}^2}{L} \\
    y_n & = & (1 - \tau_n) x_n + \tau_n z_n,\\
    z_{n+1} & = & z_n - \eta_n \nabla f(y_n), \\
    x_{n+1} & = & y_n - \gamma \nabla f(y_n).
\end{aligned}
```
# Theoretical guarantee
The following worst-case guarantee can be found in e.g., [2, Theorem 5.3]:

```math
V_{n+1} - V_n  \leqslant 0,
```

when $\gamma=\frac{1}{L}$.

# References
The potential can be found in the historical [1]; and in more recent works, e.g., [2, 3].

[[1] Y. Nesterov (1983).
A method for solving the convex programming problem with convergence rate $O(1/k^2)$.
In Dokl. akad. nauk Sssr (Vol. 269, pp. 543-547).](http://www.mathnet.ru/links/9bcb158ed2df3d8db3532aafd551967d/dan46009.pdf)

[[2] N. Bansal, A. Gupta (2019).
Potential-function proofs for gradient methods.
Theory of Computing, 15(1), 1-32.](https://arxiv.org/pdf/1712.04581.pdf)

[[3] A. d'Aspremont, D. Scieur, A. Taylor (2021).
Acceleration Methods.
Foundations and Trends in Optimization: Vol. 5, No. 1-2.](https://arxiv.org/pdf/2101.09545.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `lam`: the initial value for sequence $(\lambda_t)_t$.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_accelerated_gradient_method(1.0, 1.0, 10.0; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.0, 0.0)
```
"""
function wc_accelerated_gradient_method(L, gamma, lam; verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param; reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xn = set_initial_point!(problem)
    _gn, fn = oracle!(func, xn)
    zn = set_initial_point!(problem)


    lam_np1 = (1 + sqrt(4 * lam^2 + 1)) / 2
    tau = 1 / lam_np1
    yn = (1 - tau) * xn + tau * zn
    gyn = gradient!(func, yn)

    eta = (lam_np1^2 - lam^2) / L
    znp1 = zn - eta * gyn

    xnp1 = yn - gamma * gyn
    _gnp1, fnp1 = oracle!(func, xnp1)


    final_lyapunov = lam_np1^2 * (fnp1 - fs) + L / 2 * (znp1 - xs)^2
    init_lyapunov = lam^2 * (fn - fs) + L / 2 * (zn - xs)^2

    set_performance_metric!(problem, final_lyapunov - init_lyapunov)

    pepit_tau = solve!(problem; verbose=verbose)

    theoretical_tau = gamma == 1 / L ? 0.0 : nothing

    if verbose
        println("*** Example file: worst-case performance of accelerated gradient method for a given Lyapunov function ***")
        println("\tPEPit guarantee:\t V_(n+1) - V_n <= $(round(pepit_tau, digits=6))")
        if gamma == 1 / L
            println("\tTheoretical guarantee:\t V_(n+1) - V_n <= $(round(theoretical_tau, digits=6))")
        end
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_gradient_method(1.0, 1.0, 10.0; verbose=true)
