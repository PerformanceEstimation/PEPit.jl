using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_accelerated_gradient_convex(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_gradient_convex`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex ($\mu$ is possibly 0).

# Performance metric

This code computes a worst-case guarantee for an **accelerated gradient method**, a.k.a. **fast gradient method** [1].
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the accelerated gradient method,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
Initialize $\lambda_1=1$, $y_1=x_0$.
One iteration of accelerated gradient method is described by

```math
\begin{align}
    \text{Set: }\lambda_{t+1} & = & \frac{1 + \sqrt{4\lambda_t^2 + 1}}{2} \\
    x_{t} & = & y_t - \frac{1}{L} \nabla f(y_t),\\
    y_{t+1} & = & x_{t} + \frac{\lambda_t-1}{\lambda_{t+1}} (x_t-x_{t-1}).
\end{align}
```
# Theoretical guarantee
The following worst-case guarantee can be found in e.g., [2, Theorem 4.4]:

```math
f(x_n)-f_\star \leqslant \frac{L}{2}\frac{\|x_0-x_\star\|^2}{\lambda_n^2}.
```

# References


[[1] Y. Nesterov (1983).
A method for solving the convex programming problem with convergence rate O(1/k^2).
In Dokl. akad. nauk Sssr (Vol. 269, pp. 543-547).](http://www.mathnet.ru/links/9bcb158ed2df3d8db3532aafd551967d/dan46009.pdf)

[[2] A. Beck, M. Teboulle (2009).
A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
SIAM journal on imaging sciences, 2009, vol. 2, no 1, p. 183-202.](https://www.ceremade.dauphine.fr/~carlier/FISTA)

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
pepit_tau, theoretical_tau = wc_accelerated_gradient_convex(0, 1, 1; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.166667, 0.5)
```
"""
function wc_accelerated_gradient_convex(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    y = x0
    lam = 1.0
    lam_old = lam

    for _ in 1:n
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old^2 + 1)) / 2
        x_old = x
        x = y - 1 / L * gradient!(func, y)
        y = x + (lam_old - 1) / lam * (x - x_old)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * lam_old^2)

    mu != 0 && @warn "Momentum is tuned for non-strongly convex functions."


    if verbose
        println("*** Example file: worst-case performance of accelerated gradient method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_convex(0, 1, 1; solver=Clarabel.Optimizer, verbose=true)
