using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth.

# Performance metric

This code computes a worst-case guarantee for **gradient descent** with fixed step-size $\gamma$,
and looks for a low-dimensional worst-case example nearly achieving this worst-case guarantee.
That is, it computes the smallest possible $\tau(n, L, \gamma)$ such that the guarantee

```math
\min_{t\leqslant n} \|\nabla f(x_t)\|^2 \leqslant \tau(n, L, \gamma) (f(x_0) - f(x_n))
```

is valid, where $x_n$ is the n-th iterates obtained with the gradient method with fixed step-size.
Then, it looks for a low-dimensional nearly achieving this performance.

# Algorithm

Gradient descent is described as follows, for $t \in \{ 0, \dots, n-1\}$,

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size and.

# Theoretical guarantee

When $\gamma \leqslant \frac{1}{L}$, an empirically tight theoretical worst-case guarantee is

```math
\min_{t\leqslant n} \|\nabla f(x_t)\|^2 \leqslant \frac{4}{3}\frac{L}{n} (f(x_0) - f(x_n)),
```

see discussions in [1, page 190] and [2].

# References


[[1] Taylor, A. B. (2017). Convex interpolation and performance estimation of first-order methods for
convex optimization. PhD Thesis, UCLouvain.](https://dial.uclouvain.be/downloader/downloader.php?pid=boreal:182881&datastream=PDF_01)

[[2] H. Abbaszadehpeivasti, E. de Klerk, M. Zamani (2021). The exact worst-case convergence rate of the
gradient method with fixed step lengths for L-smooth functions. Optimization Letters, 16(6), 1649-1661.](https://arxiv.org/pdf/2104.05468v3.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent(L, gamma, 5; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.266657, 0.266667)
```
"""
function wc_gradient_descent(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, SmoothFunction, OrderedDict("L" => L))


    x0 = set_initial_point!(problem)
    g0, f0 = oracle!(func, x0)


    x = x0
    gx, fx = g0, f0


    set_performance_metric!(problem, gx^2)

    for i in 1:n
        x = x - gamma * gx

        gx, fx = oracle!(func, x)
        set_performance_metric!(problem, gx^2)
    end


    set_initial_condition!(problem, f0 - fx <= 1)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=2)


    theoretical_tau = 4 / 3 * L / n


    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-size ***")
        println("\tPEPit guarantee:\t min_i ||f'(x_i)||^2 == $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t min_i ||f'(x_i)||^2 <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
gamma = 1 / L
pepit_tau, theoretical_tau = wc_gradient_descent(L, gamma, 5; solver=Clarabel.Optimizer, verbose=true)
