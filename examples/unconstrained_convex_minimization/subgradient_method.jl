using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_subgradient_method(M, n, gamma; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_subgradient_method`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is convex and $M$-Lipschitz. This problem is a (possibly non-smooth) minimization problem.

# Performance metric

This code computes a worst-case guarantee for the **subgradient method**. That is, it computes
the smallest possible $\tau(n, M, \gamma)$ such that the guarantee

```math
\min_{0 \leqslant t \leqslant n} f(x_t) - f_\star \leqslant \tau(n, M, \gamma)
```

is valid, where $x_t$ are the iterates of the **subgradient method** after $t\leqslant n$ steps,
where $x_\star$ is a minimizer of $f$, and when $\|x_0-x_\star\|\leqslant 1$.

In short, for given values of $M$, the step-size $\gamma$ and the number of iterations $n$,
$\tau(n, M, \gamma)$ is computed as the worst-case value of
$\min_{0 \leqslant t \leqslant n} f(x_t) - f_\star$ when  $\|x_0-x_\star\| \leqslant 1$.

# Algorithm

For $t\in \{0, \dots, n-1 \}$

```math
    \begin{aligned}
        g_{t} & \in & \partial f(x_t) \\
        x_{t+1} & = & x_t - \gamma g_t
    \end{aligned}
```
# Theoretical guarantee
The **tight** bound is obtained in [1, Section 3.2.3] and [2, Eq (2)]

```math
\min_{0 \leqslant t \leqslant n} f(x_t)- f(x_\star) \leqslant \frac{M}{\sqrt{n+1}}\|x_0-x_\star\|,
```

and tightness follows from the lower complexity bound for this class of problems, e.g., [3, Appendix A].

# References
Classical references on this topic include [1, 2].

[[1] Y. Nesterov (2003).
Introductory lectures on convex optimization: A basic course.
Springer Science & Business Media.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.855&rep=rep1&type=pdf)

[[2] S. Boyd, L. Xiao, A. Mutapcic (2003).
Subgradient Methods (lecture notes).](https://web.stanford.edu/class/ee392o/subgrad_method.pdf)

[[3] Y. Drori, M. Teboulle (2016).
An optimal variant of Kelley's cutting-plane method.
Mathematical Programming, 160(1), 321-351.](https://arxiv.org/pdf/1409.2636.pdf)

# Arguments
- `M`: the Lipschitz parameter.
- `n`: number of iterations.
- `gamma`: step-size parameter.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_subgradient_method(M, n, gamma; verbose=true)
```
"""
function wc_subgradient_method(M, n, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    func = declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    gx, fx = oracle!(func, x)

    for _ in 1:n
        set_performance_metric!(problem, fx - fs)
        x = x - gamma * gx
        gx, fx = oracle!(func, x)
    end


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = M / sqrt(n + 1)

    if verbose
        println("*** Example file: worst-case performance of subgradient method ***")
        println("\tPEPit guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||")
        println("\tTheoretical guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||")
    end

    return pepit_tau, theoretical_tau
end


M = 2
n = 6
gamma = 1 / (M * sqrt(n + 1))
pepit_tau, theoretical_tau = wc_subgradient_method(M, n, gamma; verbose=true)
