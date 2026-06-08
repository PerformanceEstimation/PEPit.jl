using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_quadratics(mu, L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_quadratics`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f=\frac{1}{2} x^T Q x$ is $L$-smooth and $\mu$-strongly convex (i.e. $\mu I \preceq Q \preceq LI$).

# Performance metric

This code computes a worst-case guarantee for **gradient descent** with fixed step-size $\gamma$.
That is, it computes the smallest possible $\tau(n, \mu, L, \gamma)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, \mu, L, \gamma) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of gradient descent with fixed step-size $\gamma$, and
where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$, $\mu$, $L$, and $\gamma$, $\tau(n, L, \gamma)$ is computed as the worst-case
value of $f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

Gradient descent is described by

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size.

# Theoretical guarantee

    When $\gamma \leqslant \frac{2}{L}$ and $0 \leqslant \mu \leqslant L$,
    the **tight** theoretical conjecture can be found in [1, Equation (4.17)]:

```math
f(x_n)-f_\star \leqslant \frac{L}{2} \max\left\{\alpha(1-\alpha L\gamma)^{2n}, (1-L\gamma)^{2n} \right\} \|x_0-x_\star\|^2,
```

where $\alpha = \mathrm{proj}_{[\frac{\mu}{L},1]} \left(\frac{1}{L\gamma (2n+1)}\right)$.

# References


    [[1] N. Bousselmi, J. Hendrickx, F. Glineur  (2023).
    Interpolation Conditions for Linear Operators and applications to Performance Estimation Problems.
    arXiv preprint](https://arxiv.org/pdf/2302.08781.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent_quadratics(mu, L, 1 / L, 4; verbose=true)
```
"""
function wc_gradient_descent_quadratics(mu, L, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexQuadraticFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for _ in 1:n
        x = x - gamma * gradient!(func, x)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    t = 1 / (L * gamma * (2 * n + 1))
    if t < mu / L
        alpha = mu / L
    elseif t > 1
        alpha = 1
    else
        alpha = t
    end

    theoretical_tau = 0.5 * L * max(alpha * (1 - alpha * L * gamma)^(2 * n), (1 - L * gamma)^(2 * n))

    if verbose
        println("*** Example file: worst-case performance of gradient descent on quadratics with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


L = 3.0
mu = 0.3
pepit_tau, theoretical_tau = wc_gradient_descent_quadratics(mu, L, 1 / L, 4; verbose=true)
