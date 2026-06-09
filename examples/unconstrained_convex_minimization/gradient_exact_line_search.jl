using PEPit
using OrderedCollections

@doc raw"""
    wc_gradient_exact_line_search(L, mu, n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_exact_line_search`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex.

# Performance metric

This code computes a worst-case guarantee for the **gradient descent** (GD) with **exact linesearch** (ELS).
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, L, \mu) (f(x_0) - f_\star)
```

is valid, where $x_n$ is the output of the GD with ELS,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$f(x_n)-f_\star$ when $f(x_0) - f_\star \leqslant 1$.

# Algorithm

GD with ELS can be written as

```math
x_{t+1} = x_t - \gamma_t \nabla f(x_t)
```

with $\gamma_t = \arg\min_{\gamma} f \left( x_t - \gamma \nabla f(x_t) \right)$.

# Theoretical guarantee
The **tight** worst-case guarantee for GD with ELS, obtained in [1, Theorem 1.2], is

```math
f(x_n) - f_\star \leqslant \left(\frac{L-\mu}{L+\mu}\right)^{2n} (f(x_0) - f_\star).
```

# References
The detailed approach (based on convex relaxations) is available in [1],
along with theoretical bound.

[[1] E. De Klerk, F. Glineur, A. Taylor (2017).
On the worst-case complexity of the gradient method with exact line search for smooth strongly convex functions.
Optimization Letters, 11(7), 1185-1199.](https://link.springer.com/content/pdf/10.1007/s11590-016-1087-4.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_exact_line_search(1.0, 0.1, 2; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.448125, 0.448125)
```
"""
function wc_gradient_exact_line_search(L, mu, n; verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param; reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x = set_initial_point!(problem)
    gx, f0 = oracle!(func, x)


    set_initial_condition!(problem, f0 - fs <= 1)


    fx = f0
    for i in 1:n
        x, gx, fx = exact_linesearch_step!(x, func, [gx])
    end


    set_performance_metric!(problem, fx - fs)

    pepit_tau = solve!(problem; verbose=verbose)
    theoretical_tau = ((L - mu) / (L + mu))^(2 * n)

    if verbose
        println("*** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_exact_line_search(1.0, 0.1, 2; verbose=true)
