using PEPit, OrderedCollections, Clarabel


function phi(mu, idx)
    if idx == -1
        return 0
    end
    return ((1 + 2 * mu)^(2 * idx + 2) - 1) / ((1 + 2 * mu)^2 - 1)
end

@doc raw"""
    wc_optimal_strongly_monotone_proximal_point(n, mu; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_optimal_strongly_monotone_proximal_point`.

Compute a worst-case guarantee for optimal strongly monotone proximal point.

# Performance metric

This code computes a worst-case guarantee for the squared final residual

```math
\|y_{n-1} - x_n\|^2,
```

under the normalization $\|x_0-x_\star\|^2 \leqslant 1$, where $x_\star$ is a zero
of the strongly monotone operator $A$.

# Algorithm

The method applies a unit-stepsize proximal point oracle for a $\mu$-strongly
monotone operator and then extrapolates the next query point. With

```math
\phi_i(\mu) =
\begin{cases}
0, & i=-1,\\
\frac{(1+2\mu)^{2i+2}-1}{(1+2\mu)^2-1}, & i\geq 0,
\end{cases}
```

the Julia implementation uses

```math
\begin{aligned}
x_{i+1} &= (I + A)^{-1}(y_i),\\
y_{i+1} &= x_{i+1}
    + \frac{\phi_i(\mu)-1}{\phi_{i+1}(\mu)}(x_{i+1}-x_i)
    - \frac{2\mu\phi_i(\mu)}{\phi_{i+1}(\mu)}(y_i-x_{i+1})\\
&\quad
    + \frac{(1+2\mu)\phi_{i-1}(\mu)}{\phi_{i+1}(\mu)}(y_{i-1}-x_i).
\end{aligned}
```

# Theoretical guarantee

The reference value computed by the example is

```math
\left(\frac{2\mu}{(1+2\mu)^n-1}\right)^2.
```

# References

The detailed approach and tight bound are available in [1].

[[1] J. Park, E. Ryu (2022).
Exact Optimal Accelerated Complexity for Fixed-Point Iterations.
In 39th International Conference on Machine Learning (ICML).](https://proceedings.mlr.press/v162/park22c/park22c.pdf)

# Arguments
- `n`: number of iterations.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value computed by PEPit.jl.
- `theoretical_tau`: reference theoretical value when the example provides one.

# Julia usage
```julia
wc_optimal_strongly_monotone_proximal_point(10, 0.05; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.003937, 0.003937)
```
"""
function wc_optimal_strongly_monotone_proximal_point(n, mu; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, StronglyMonotoneOperator, OrderedDict("mu" => mu))


    xs = stationary_point!(A)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x, y, y_prv = x0, x0, x0
    for i in 0:(n - 1)
        x_nxt, _, _ = proximal_step!(y, A, 1)
        y_nxt = x_nxt + (phi(mu, i) - 1) / phi(mu, i + 1) * (x_nxt - x) - 2 * mu * phi(mu, i) / phi(mu, i + 1) * (
                y - x_nxt) + (1 + 2 * mu) * phi(mu, i - 1) / phi(mu, i + 1) * (y_prv - x)
        x, y_prv, y = x_nxt, y, y_nxt
    end


    set_performance_metric!(problem, (y_prv - x)^2)


    pepit_verbose = verbose >= 0
    pepit_tau = solve!(problem; solver=solver, verbose=pepit_verbose)


    theoretical_tau = (2 * mu / ((1 + 2 * mu)^n - 1))^2


    if verbose != -1
        println("*** Example file: worst-case performance of Optimal Strongly-monotone Proximal Point Method ***")
        println("\tPEPit guarantee:\t ||AxN||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||AxN||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_optimal_strongly_monotone_proximal_point(10, 0.05; verbose=true)
