using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_subgradient_method_rsi_eb(mu, L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_subgradient_method_rsi_eb`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ verifies the "lower" restricted secant inequality ($\mu-\text{RSI}^-$)
and the "upper" error bound ($L-\text{EB}^+$) [1].

# Performance metric

This code computes a worst-case guarantee for **gradient descent** with fixed step-size $\gamma$.
That is, it computes the smallest possible $\tau(n, \mu, L, \gamma)$ such that the guarantee

```math
\| x_n - x_\star \|^2 \leqslant \tau(n, \mu, L, \gamma) \| x_0 - x_\star \|^2
```

is valid, where $x_n$ is the output of gradient descent with fixed step-size $\gamma$, and
where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$, $L$, and $\gamma$,
$\tau(n, \mu, L, \gamma)$ is computed as the worst-case value of
$\| x_n - x_\star \|^2$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

Sub-gradient descent is described by

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size.

# Theoretical guarantee

The **tight** theoretical guarantee can be found in [1, Prop 1] (upper bound) and [1, Theorem 2] (lower bound):

```math
\| x_n - x_\star \|^2 \leqslant (1 - 2\gamma\mu + L^2 \gamma^2)^n \|x_0-x_\star\|^2.
```

# References

Definition and convergence guarantees can be found in [1].

[[1] C. Guille-Escuret, B. Goujaud, A. Ibrahim, I. Mitliagkas (2022).
Gradient Descent Is Optimal Under Lower Restricted Secant Inequality And Upper Error Bound.](https://arxiv.org/pdf/2203.00342.pdf)

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
pepit_tau, theoretical_tau = wc_subgradient_method_rsi_eb(mu, L, mu / L^2, 4; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.960596, 0.960596)
```
"""
function wc_subgradient_method_rsi_eb(mu, L, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, RsiEbFunction, param)


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for _ in 1:n
        x = x - gamma * gradient!(func, x)
    end


    set_performance_metric!(problem, (x - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 - 2 * gamma * mu + gamma^2 * L^2)^n

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


mu = 0.1
L = 1.0
pepit_tau, theoretical_tau = wc_subgradient_method_rsi_eb(mu, L, mu / L^2, 4; solver=Clarabel.Optimizer, verbose=true)
