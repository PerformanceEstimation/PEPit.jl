using PEPit, OrderedCollections, Clarabel


@doc raw"""
    wc_accelerated_proximal_gradient_simplified(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_proximal_gradient_simplified`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f(x) + h(x)\},
```

where $f$ is $L$-smooth and $\mu$-strongly convex,
and where $h$ is closed convex and proper.

# Performance metric

This code computes a worst-case guarantee for the **accelerated proximal gradient** method,
also known as **fast proximal gradient (FPGM)** method.
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
F(x_n) - F(x_\star) \leqslant \tau(n, L, \mu) \|x_0 - x_\star\|^2,
```

is valid, where $x_n$ is the output of the **accelerated proximal gradient** method,
and where $x_\star$ is a minimizer of $F$.

In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$F(x_n) - F(x_\star)$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
Accelerated proximal gradient is described as follows, for $t \in \{ 0, \dots, n-1\}$,

```math
\begin{aligned}
    x_{t+1} & = & \arg\min_x \left\{h(x)+\frac{L}{2}\|x-\left(y_{t} - \frac{1}{L} \nabla f(y_t)\right)\|^2 \right\}, \\
    y_{t+1} & = & x_{t+1} + \frac{i}{i+3} (x_{t+1} - x_{t}),
\end{aligned}
```
where $y_{0} = x_0$.

# Theoretical guarantee
A **tight** (empirical) worst-case guarantee for FPGM is obtained in
[1, method FPGM1 in Sec. 4.2.1, Table 1 in sec 4.2.2], for $\mu=0$:

```math
F(x_n) - F_\star \leqslant \frac{2 L}{n^2+5n+2} \|x_0 - x_\star\|^2,
```

which is attained on simple one-dimensional constrained linear optimization problems.

# References


[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex optimization.
SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_accelerated_proximal_gradient_simplified(0.0, 1.0, 4; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.052632, 0.052632)
```
"""
function wc_accelerated_proximal_gradient_simplified(mu, L, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    h = declare_function!(problem, ConvexFunction, OrderedDict())
    F = f + h


    xs = stationary_point!(F)
    Fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    y = x0
    local hx_new::Expression
    for i in 0:(n-1)

        x_old = x_new


        x_new, _, hx_new = proximal_step!(y - 1 / L * gradient!(f, y), h, 1 / L)


        y = x_new + i / (i + 3) * (x_new - x_old)
    end


    set_performance_metric!(problem, (value!(f, x_new) + hx_new) - Fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 * L / (n^2 + 5 * n + 2)
    if mu != 0
        println("Warning: momentum is tuned for non-strongly convex functions.")
    end


    if verbose
        println("*** Example file: worst-case performance of the Accelerated Proximal Gradient Method in function values***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_proximal_gradient_simplified(0.0, 1.0, 4; verbose=true)
