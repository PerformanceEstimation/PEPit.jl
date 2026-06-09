using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, n, alpha=0; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_difference_of_convex_algorithm`.

Consider the minimization problem

```math
F_\star \triangleq \min_x f_1(x)-f_2(x),
```

where $f_1$ and $f_2$ are convex functions, respectively $L_1$-smooth and
$\mu_1$-strongly convex and $L_2$-smooth and $\mu_2$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for **DCA** (difference-of-convex algorithm, also known as the
convex-concave procedure). That is, it computes the smallest possible $\tau(n, \mu_1, L_1,\mu_2, L_2)$
such that the guarantee

```math
\min_{t\leqslant n} \|\nabla f_1(x_t)-\nabla f_2(x_t)\|^2 \leqslant \tau(n, \mu_1, L_1,\mu_2, L_2) (f_1(x_0)-f_2(x_0)-F_\star)
```

is valid, where $x_n$ is the n-th iterates obtained with DCA.

# Algorithm

DCA is described as follows, for $t \in \{ 0, \dots, n-1\}$,

```math
x_{t+1} \in \mathrm{argmin}_x\,\{ f_1(x) - \langle \nabla f_2(x_t), x\rangle\},
```


# Theoretical guarantee
The results are compared with [1, Theorem 3];
a more complete picture can be obtained from [2], also by possibly allowing for non-convex functions
$f_1$ and $f_2$ (i.e., possibly negative values for $\mu_1$, $\mu_2$).

# References


[[1] H. Abbaszadehpeivasti, E. de Klerk, M. Zamani (2021).
On the rate of convergence of the difference-of-convex algorithm (DCA).
Journal of Optimization Theory and Applications, 202(1), 475-496.](https://arxiv.org/pdf/2109.13566)

[[2] T. Rotaru, P. Patrinos, F. Glineur (2025).
Tight Analysis of Difference-of-Convex Algorithm (DCA) Improves Convergence Rates for Proximal Gradient Descent.
Journal of Optimization Theory and Applications, 202(1), 475-496.](https://arxiv.org/pdf/2503.04486)

# Arguments
- `mu1`: strong convexity parameter for f1.
- `mu2`: strong convexity parameter for f2.
- `L1`: smoothness parameter for f1.
- `L2`: smoothness parameter for f2.
- `n`: number of iterations.
- `alpha`: algorithm parameter used in the update rule.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: reference theoretical value [1, Theorem 3].

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, 5, 0; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.491131, 0.491131)
```
"""
function wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, n, alpha=0; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu1, "L" => L1))
    f2 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu2, "L" => L2))
    F = f1 - f2


    xs = stationary_point!(F)
    Fs = value!(F, xs)


    x0 = set_initial_point!(problem)

    x = x0
    g1x = gradient!(f1, x0)
    g2x = gradient!(f2, x0)


    set_initial_condition!(problem, value!(f1, x) - value!(f2, x) - Fs <= 1)

    for i in 1:n
        set_performance_metric!(problem, (g1x - g2x)^2)
        add_constraint!(problem, Fs <= value!(f1, x) - value!(f2, x) - 1 / 2 / (L1 - mu2) * (g1x - g2x)^2)
        y, _, _ = shifted_optimization_step!(g2x, f1)
        x = (1 + alpha) * y - alpha * x
        g1x, f1x = oracle!(f1, x)
        g2x, f1x = oracle!(f2, x)
    end

    set_performance_metric!(problem, (g1x - g2x)^2)
    add_constraint!(problem, Fs <= value!(f1, x) - value!(f2, x) - 1 / (2 * (L1 - mu2)) * (g1x - g2x)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if alpha == 0
        Delta = 1
        A = 2 * (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1))
        B = L1 + L2 + mu1 * (L1 / L2 - 3) * (L1 >= L2) + mu2 * (L2 / L1 - 3) * (L2 > L1)
        C = (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1)) / (L1 - mu2)
        theoretical_tau = A * Delta / (B * n + C)
    else
        theoretical_tau = nothing
    end

    if verbose
        println("*** Example file: worst-case performance of DCA ***")
        println("\tPEPit guarantee:\t min_i ||f'(x_i)||^2 <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t min_i ||f'(x_i)||^2 <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


L1, L2, mu1, mu2 = 2.0, 5.0, 0.15, 0.1
pepit_tau, theoretical_tau = wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, 5, 0; solver=Clarabel.Optimizer, verbose=true)
