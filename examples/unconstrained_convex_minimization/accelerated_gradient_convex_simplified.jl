using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_accelerated_gradient_convex_simplified(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_gradient_convex_simplified`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex ($\mu$ is possibly 0).

# Performance metric

This code computes a worst-case guarantee for an **accelerated gradient method**, a.k.a. **fast gradient method**
with a set of classical slightly simplified sets of coefficients compared to the original [1].
That is, the code computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the accelerated gradient method below,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

The accelerated gradient method of this example is provided by

```math
    \begin{aligned}
        x_{t+1} & = & y_t - \frac{1}{L} \nabla f(y_t) \\
        y_{t+1} & = & x_{t+1} + \frac{t-1}{t+2} (x_{t+1} - x_t).
    \end{aligned}
```
# Theoretical guarantee

When $\mu=0$, a tight **empirical** guarantee can be found in [2, Table 1]:

```math
f(x_n)-f_\star \leqslant \frac{2L\|x_0-x_\star\|^2}{n^2 + 5 n + 6},
```

where tightness is obtained on some Huber loss functions.

# References


[[1] Y. Nesterov (1983).
A method for solving the convex programming problem with convergence rate O(1/k^2).
In Dokl. akad. nauk Sssr (Vol. 269, pp. 543-547).](http://www.mathnet.ru/links/9bcb158ed2df3d8db3532aafd551967d/dan46009.pdf)

[[2] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex optimization.
SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

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
pepit_tau, theoretical_tau = wc_accelerated_gradient_convex_simplified(0, 1, 1; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_accelerated_gradient_convex_simplified(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    y = x0
    for i in 1:n
        ipy = i - 1
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        y = x_new + ipy / (ipy + 3) * (x_new - x_old)
    end


    set_performance_metric!(problem, value!(func, x_new) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 * L / (n^2 + 5 * n + 6)
    mu != 0 && @warn "Momentum is tuned for non-strongly convex functions."


    if verbose
        println("*** Example file: worst-case performance of accelerated gradient method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_convex_simplified(0, 1, 1; solver=Clarabel.Optimizer, verbose=true)
