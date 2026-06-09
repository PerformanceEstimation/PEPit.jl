using PEPit, OrderedCollections, Clarabel, OffsetArrays


@doc raw"""
    wc_accelerated_douglas_rachford_splitting(mu, L, alpha, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_douglas_rachford_splitting`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f_1(x) + f_2(x)\},
```

where $f_1$ is closed convex and proper, and $f_2$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for **accelerated Douglas-Rachford**. That is, it computes
the smallest possible $\tau(n, L, \mu, \alpha)$ such that the guarantee

```math
F(y_n) - F(x_\star) \leqslant \tau(n,L,\mu,\alpha) \|w_0 - w_\star\|^2
```

is valid, $\alpha$ is a parameter of the method, and where $y_n$ is the output
of the accelerated Douglas-Rachford Splitting method, where $x_\star$ is a minimizer of $F$,
and $w_\star$ defined such that

```math
x_\star = \mathrm{prox}_{\alpha f_2}(w_\star)
```

is an optimal point.

In short, for given values of $n$, $L$, $\mu$, $\alpha$,
$\tau(n, L, \mu, \alpha)$ is computed as the worst-case value of $F(y_n)-F_\star$
when $\|w_0 - w_\star\|^2 \leqslant 1$.

# Algorithm

The accelerated Douglas-Rachford splitting is described in [1, Section 4]. For $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        x_{t} & = & \mathrm{prox}_{\alpha f_2} (u_t),\\
        y_{t} & = & \mathrm{prox}_{\alpha f_1}(2x_t-u_t),\\
        w_{t+1} & = & u_t + \theta (y_t-x_t),\\
        u_{t+1} & = & \left\{\begin{array}{ll} w_{t+1}+\frac{t-1}{t+2}(w_{t+1}-w_t)\, & \text{if } t >1,\\
        w_{t+1} & \text{otherwise.} \end{array}\right.
    \end{aligned}
```
# Theoretical guarantee

There is no known worst-case guarantee for this method beyond quadratic minimization.
For quadratics, an **upper** bound on is provided by [1, Theorem 5]:

```math
F(y_n) - F_\star \leqslant \frac{2}{\alpha \theta (n + 3)^ 2} \|w_0-w_\star\|^2,
```

when $\theta=\frac{1-\alpha L}{1+\alpha L}$ and $\alpha < \frac{1}{L}$.

# References

An analysis of the accelerated Douglas-Rachford splitting is available in [1, Theorem 5] for when the convex
minimization problem is quadratic.

[[1] P. Patrinos, L. Stella, A. Bemporad (2014).
Douglas-Rachford splitting: Complexity estimates and accelerated variants.
In 53rd IEEE Conference on Decision and Control (CDC).](https://arxiv.org/pdf/1407.6723.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `alpha`: algorithm parameter used in the update rule.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value (upper bound for quadratics; not directly comparable).

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_accelerated_douglas_rachford_splitting(0.1, 1.0, 0.9, 2; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.192915, 1.688889)
```
"""
function wc_accelerated_douglas_rachford_splitting(mu, L, alpha, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    func1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)
    func2 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L); reuse_gradient=true)


    func = func1 + func2


    xs = stationary_point!(func)
    fs = value!(func, xs)
    g1s, _ = oracle!(func1, xs)
    g2s, _ = oracle!(func2, xs)


    x0 = set_initial_point!(problem)


    theta = (1 - alpha * L) / (1 + alpha * L)


    ws = xs + alpha * g2s
    set_initial_condition!(problem, (ws - x0)^2 <= 1)


    x = OffsetVector([x0 for _ in 0:(n - 1)], 0:(n - 1))
    w = OffsetVector([x0 for _ in 0:n], 0:n)
    u = OffsetVector([x0 for _ in 0:n], 0:n)


    local y
    local fy


    for i in 0:(n - 1)

        x[i], _, _ = proximal_step!(u[i], func2, alpha)


        y, _, fy = proximal_step!(2 * x[i] - u[i], func1, alpha)


        w[i + 1] = u[i] + theta * (y - x[i])


        if i >= 1
            u[i + 1] = w[i + 1] + (i - 1) / (i + 2) * (w[i + 1] - w[i])
        else
            u[i + 1] = w[i + 1]
        end
    end


    set_performance_metric!(problem, value!(func2, y) + fy - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if alpha < 1 / L
        theoretical_tau = 2 / (alpha * theta * (n + 3)^2)
    else
        theoretical_tau = nothing
    end


    if verbose
        println("*** Example file: worst-case performance of the Accelerated Douglas Rachford Splitting in function values ***")
        println("\tPEPit guarantee:\t\t\t F(y_n)-F_* <= $(round(pepit_tau, digits=6)) ||x0 - ws||^2")
        if alpha < 1 / L
            println("\tTheoretical guarantee for quadratics:\t F(y_n)-F_* <= $(round(theoretical_tau, digits=6)) ||x0 - ws||^2")
        end
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_douglas_rachford_splitting(0.1, 1.0, 0.9, 2; verbose=true)
