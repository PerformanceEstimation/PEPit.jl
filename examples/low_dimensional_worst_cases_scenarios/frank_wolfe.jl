using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_frank_wolfe(L, D, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_frank_wolfe`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f_1(x) + f_2(x)\},
```

where $f_1$ is $L$-smooth and convex
and where $f_2$ is a convex indicator function on $\mathcal{D}$ of diameter at most $D$.

# Performance metric

This code computes a worst-case guarantee for the **conditional gradient** method, aka **Frank-Wolfe** method,
and looks for a low-dimensional worst-case example nearly achieving this worst-case guarantee using
$12$ iterations of the logdet heuristic.
That is, it computes the smallest possible $\tau(n, L)$ such that the guarantee

```math
F(x_n) - F(x_\star) \leqslant \tau(n, L) D^2,
```

is valid, where $x_n$ is the output of the **conditional gradient** method,
and where $x_\star$ is a minimizer of $F$.
In short, for given values of $n$ and $L$, $\tau(n, L)$ is computed as the worst-case value of
$F(x_n) - F(x_\star)$ when $D \leqslant 1$. Then, it looks for a low-dimensional nearly achieving this
performance.

# Algorithm


This method was first presented in [1]. A more recent version can be found in, e.g., [2, Algorithm 1].
For $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        y_t & = & \arg\min_{s \in \mathcal{D}} \langle s \mid \nabla f_1(x_t) \rangle, \\
        x_{t+1} & = & \frac{t}{t + 2} x_t + \frac{2}{t + 2} y_t.
    \end{aligned}
```
# Theoretical guarantee


An **upper** guarantee obtained in [2, Theorem 1] is

```math
F(x_n) - F(x_\star) \leqslant \frac{2L D^2}{n+2}.
```

# References
The algorithm is presented in, among others, [1, 2]. The logdet heuristic is presented in [3].

[1] M. Frank, P. Wolfe (1956).
An algorithm for quadratic programming.
Naval research logistics quarterly, 3(1-2), 95-110.

[[2] M. Jaggi (2013). Revisiting Frank-Wolfe: Projection-free sparse convex optimization.
In 30th International Conference on Machine Learning (ICML).](http://proceedings.mlr.press/v28/jaggi13.pdf)

[[3] F. Maryam, H. Hindi, S. Boyd (2003). Log-det heuristic for matrix rank minimization with applications to Hankel
and Euclidean distance matrices. American Control Conference (ACC).](https://web.stanford.edu/~boyd/papers/pdf/rank_min_heur_hankel.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `D`: diameter of $f_2$.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_frank_wolfe(1.0, 1.0, 10; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.07828, 0.166667)
```
"""
function wc_frank_wolfe(L, D, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func1 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L))
    func2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))

    func = func1 + func2


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    _ = value!(func1, x0)
    _ = value!(func2, x0)


    x = x0
    for i in 0:(n-1)
        g = gradient!(func1, x)
        y, _, _ = linear_optimization_step!(g, func2)
        lam = 2 / (i + 2)
        x = (1 - lam) * x + lam * y
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=12)


    theoretical_tau = 2 * L * D^2 / (n + 2)


    if verbose
        println("*** Example file: worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* == $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_frank_wolfe(1.0, 1.0, 10; solver=Clarabel.Optimizer, verbose=true)
