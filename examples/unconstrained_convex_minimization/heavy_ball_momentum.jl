using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_heavy_ball_momentum(mu, L, alpha, beta, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_heavy_ball_momentum`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for the **Heavy-ball (HB)** method, aka **Polyak momentum** method.
That is, it computes the smallest possible $\tau(n, L, \mu, \alpha, \beta)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu, \alpha, \beta) (f(x_0) - f_\star)
```

is valid, where $x_n$ is the output of the **Heavy-ball (HB)** method,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu, \alpha, \beta)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $f(x_0) - f_\star \leqslant 1$.

# Algorithm


```math
x_{t+1} = x_t - \alpha \nabla f(x_t) + \beta (x_t-x_{t-1})
```

    with

```math
\alpha \in (0, \frac{1}{L}]
```

    and

```math
\beta = \sqrt{(1 - \alpha \mu)(1 - L \alpha)}
```

# Theoretical guarantee


The **upper** guarantee obtained in [2, Theorem 4] is

```math
f(x_n) - f_\star \leqslant (1 - \alpha \mu)^n (f(x_0) - f_\star).
```

# References
This methods was first introduce in [1, Section 2],
and convergence upper bound was proven in [2, Theorem 4].

[[1] B.T. Polyak (1964).
Some methods of speeding up the convergence of iteration method.
URSS Computational Mathematics and Mathematical Physics.](https://www.sciencedirect.com/science/article/pii/0041555364901375)

[[2] E. Ghadimi, H. R. Feyzmahdavian, M. Johansson (2015).
Global convergence of the Heavy-ball method for convex optimization.
European Control Conference (ECC).](https://arxiv.org/pdf/1412.7457.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `alpha`: algorithm parameter used in the update rule.
- `beta`: operator or algorithm parameter used in the model.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_heavy_ball_momentum(mu, L, alpha, beta, 2; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.753493, 0.9025)
```
"""
function wc_heavy_ball_momentum(mu, L, alpha, beta, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    f0 = value!(func, x0)


    set_initial_condition!(problem, (f0 - fs) <= 1)


    x_new = x0
    x_old = x0
    for _ in 1:n
        x_next = x_new - alpha * gradient!(func, x_new) + beta * (x_new - x_old)
        x_old = x_new
        x_new = x_next
    end


    set_performance_metric!(problem, value!(func, x_new) - fs)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)
    theoretical_tau = (1 - alpha * mu)^n

    if verbose
        println("*** Example file: worst-case performance of the Heavy-Ball method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0) - f(x_*))")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0) - f(x_*))")
    end

    return pepit_tau, theoretical_tau
end

mu = 0.1
L = 1.0
alpha = 1 / (2 * L)
beta = sqrt((1 - alpha * mu) * (1 - L * alpha))
pepit_tau, theoretical_tau = wc_heavy_ball_momentum(mu, L, alpha, beta, 2; verbose=true)
