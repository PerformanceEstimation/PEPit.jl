using PEPit
using OrderedCollections
using OffsetArrays

@doc raw"""
    wc_douglas_rachford_splitting(L, alpha, theta, n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_douglas_rachford_splitting`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f_1(x)+f_2(x) \}
```

where $f_1(x)$ is is convex, closed and proper , and $f_2$ is $L$-smooth.
Both proximal operators are assumed to be available.

# Performance metric

This code computes a worst-case guarantee for the **Douglas Rachford Splitting (DRS)** method.
That is, it computes the smallest possible $\tau(n, L, \alpha, \theta)$ such that the guarantee

```math
F(y_n) - F(x_\star) \leqslant \tau(n, L, \alpha, \theta) \|x_0 - x_\star\|^2.
```

is valid, where it is known that $x_k$ and $y_k$ converge to $x_\star$, but not $w_k$
(see definitions in the section # Algorithm
). Hence we require the initial condition on $x_0$
(arbitrary choice, partially justified by the fact we choose $f_2$ to be the smooth function).

Note that $y_n$ is feasible as it
has a finite value for $f_1$ (output of the proximal operator on $f_1$) and as $f_2$ is smooth.

# Algorithm


Our notations for the DRS method are as follows, for $t \in \{0, \dots, n-1\}$,

```math
    \begin{aligned}
        x_t & = & \mathrm{prox}_{\alpha f_2}(w_t), \\
        y_t & = & \mathrm{prox}_{\alpha f_1}(2x_t - w_t), \\
        w_{t+1} & = & w_t + \theta (y_t - x_t).
    \end{aligned}
```
This description can be found in [1, Section 7.3].

# Theoretical guarantee
We compare the output with that of PESTO [2] for when $0\leqslant n \leqslant 10$
in the case where $\alpha=\theta=L=1$.

# References


[[1] E. Ryu, S. Boyd (2016).
A primer on monotone operator methods.
Applied and Computational Mathematics 15(1), 3-43.](https://web.stanford.edu/~boyd/papers/pdf/monotone_primer.pdf)

[[2] A. Taylor, J. Hendrickx, F. Glineur (2017).
Performance Estimation Toolbox (PESTO): automated worst-case analysis of first-order optimization methods.
In 56th IEEE Conference on Decision and Control (CDC).](https://github.com/AdrienTaylor/Performance-Estimation-Toolbox)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `alpha`: algorithm parameter used in the update rule.
- `theta`: relaxation or averaging parameter used in the update rule.
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
PEPit_tau, theoretical_tau = wc_douglas_rachford_splitting(1.0, 1.0, 1.0, 9; verbose=true)
# Returns approximately: (PEPit_tau, theoretical_tau) = (0.027792, 0.0278)
```
"""
function wc_douglas_rachford_splitting(L, alpha, theta, n; verbose=true)

    problem = PEP()


    f1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)
    f2 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)
    F  = f1 + f2


    xs = stationary_point!(F)
    fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    iters = 0:n-1
    x = OffsetVector(fill(x0, n), iters)
    w = OffsetVector(fill(x0, n + 1), 0:n)


    local y::Point
    local fy::Expression


    for i in iters
        xi, _, _ = proximal_step!(w[i], f2, alpha)
        x[i] = xi
        y, _, fy = proximal_step!(2 * x[i] - w[i], f1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])
    end


    set_initial_condition!(problem, (x[0] - xs)^2 <= 1)


    set_performance_metric!(problem, value!(f2, y) + fy - fs)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = nothing
    if theta == 1 && alpha == 1 && L == 1 && 1 <= n <= 10
        pesto_tau = OffsetVector([1/4, 0.1273, 0.0838, 0.0627, 0.0501,
                                  0.0417, 0.0357, 0.0313, 0.0278, 0.0250], 0:9)
        theoretical_tau = pesto_tau[n - 1]
    end

    if verbose
        println("*** Example file: worst-case performance of the Douglas–Rachford Splitting in function values ***")
        println("\tPEPit guarantee:\t f(y_n)-f_* <= $(round(PEPit_tau, digits=4)) ||x[0] - xs||^2")
        if theoretical_tau !== nothing
            println("\tTheoretical guarantee:\t f(y_n)-f_* <= $(round(theoretical_tau, digits=4)) ||x[0] - xs||^2")
        end
    end

    return PEPit_tau, theoretical_tau
end


PEPit_tau, theoretical_tau = wc_douglas_rachford_splitting(1.0, 1.0, 1.0, 9; verbose=true)
