using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_triple_momentum(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_triple_momentum`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for **triple momentum method** (TMM).
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the TMM, and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$ and $\mu$, $\tau(n, L, \mu)$ is computed
as the worst-case value of $f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.


# Algorithm


For $t \in \{ 1, \dots, n\}$

```math
    \begin{aligned}
       \xi_{t+1} &&= (1 + \beta)  \xi_{t} - \beta  \xi_{t-1} - \alpha \nabla f(y_t) \\
       y_{t} &&= (1+\gamma ) \xi_{t} -\gamma \xi_{t-1} \\
       x_{t} && = (1 + \delta)  \xi_{t} - \delta \xi_{t-1}
    \end{aligned}
```
with

```math
    \begin{aligned}
        \kappa &&= \frac{L}{\mu} , \quad \rho = 1- \frac{1}{\sqrt{\kappa}}\\
        (\alpha, \beta, \gamma,\delta) && = \left(\frac{1+\rho}{L}, \frac{\rho^2}{2-\rho},
        \frac{\rho^2}{(1+\rho)(2-\rho)}, \frac{\rho^2}{1-\rho^2}\right)
    \end{aligned}
```
and

```math
    \begin{aligned}
        \xi_{0} = x_0 \\
        \xi_{1} = x_0 \\
        y = x_0
    \end{aligned}
```
# Theoretical guarantee

A theoretical **upper** (empirically tight) bound can be found in [1, Theorem 1, eq. 4]:

```math
f(x_n)-f_\star \leqslant \frac{\rho^{2(n+1)} L \kappa}{2}\|x_0 - x_\star\|^2.
```

# References

The triple momentum method was discovered and analyzed in [1].

[[1] Van Scoy, B., Freeman, R. A., Lynch, K. M. (2018).
The fastest known globally convergent first-order method for minimizing strongly convex functions.
IEEE Control Systems Letters, 2(1), 49-54.](http://www.optimization-online.org/DB_FILE/2017/03/5908.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_triple_momentum(0.1, 1.0, 4; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_triple_momentum(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    kappa = L / mu
    rho = (1 - 1 / sqrt(kappa))
    alpha = (1 + rho) / L
    beta = rho^2 / (2 - rho)
    gamma = rho^2 / (1 + rho) / (2 - rho)
    delta = rho^2 / (1 - rho^2)


    x_old = x0
    x_new = x0
    y = x0
    local x
    for _ in 1:n
        x_inter = (1 + beta) * x_new - beta * x_old - alpha * gradient!(func, y)
        y = (1 + gamma) * x_inter - gamma * x_new
        x = (1 + delta) * x_inter - delta * x_new
        x_new, x_old = x_inter, x_new
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = rho^(2 * n) * L / 2 * kappa

    if verbose
        println("*** Example file: worst-case performance of the Triple Momentum Method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0-x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0-x_*||^2")
    end

    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_triple_momentum(0.1, 1.0, 4; solver=Clarabel.Optimizer, verbose=true)
