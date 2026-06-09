using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_heavy_ball_momentum_qg_convex(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_heavy_ball_momentum_qg_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is quadratically upper bounded ($\text{QG}^+$ [2]), i.e.
$\forall x, f(x) - f_\star \leqslant \frac{L}{2} \|x-x_\star\|^2$, and convex.

# Performance metric

This code computes a worst-case guarantee for the **Heavy-ball (HB)** method, aka **Polyak momentum** method.
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the **Heavy-ball (HB)** method,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$ and $L$,
$\tau(n, L)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm


This method is described in [1]

```math
x_{t+1} = x_t - \alpha_t \nabla f(x_t) + \beta_t (x_t-x_{t-1})
```

    with

```math
\alpha_t = \frac{1}{L} \frac{1}{t+2}
```

    and

```math
\beta_t = \frac{t}{t+2}
```

# Theoretical guarantee


The **tight** guarantee obtained in [2, Theorem 2.3] (lower) and [2, Theorem 2.4] (upper) is

```math
f(x_n) - f_\star \leqslant \frac{L}{2}\frac{1}{n+1} \|x_0 - x_\star\|^2.
```

# References

This methods was first introduce in [1, section 3],
and convergence **tight** bound was proven in [2, Theorem 2.3] (lower) and [2, Theorem 2.4] (upper).

[[1] E. Ghadimi, H. R. Feyzmahdavian, M. Johansson (2015).
Global convergence of the Heavy-ball method for convex optimization.
European Control Conference (ECC).](https://arxiv.org/pdf/1412.7457.pdf)

[[2] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
Optimal first-order methods for convex functions with a quadratic upper bound.](https://arxiv.org/pdf/2205.15033.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_heavy_ball_momentum_qg_convex(1, 5; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.083333, 0.083333)
```
"""
function wc_heavy_ball_momentum_qg_convex(L, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, ConvexQGFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    x_old = x0
    for t in 0:(n - 1)
        x_next = x_new - 1 / (L * (t + 2)) * gradient!(func, x_new) + t / (t + 2) * (x_new - x_old)
        x_old = x_new
        x_new = x_next
    end


    set_performance_metric!(problem, value!(func, x_new) - fs)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)
    theoretical_tau = L / (2 * (n + 1))

    if verbose
        println("*** Example file: worst-case performance of the Heavy-Ball method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_heavy_ball_momentum_qg_convex(1, 5; verbose=true)
