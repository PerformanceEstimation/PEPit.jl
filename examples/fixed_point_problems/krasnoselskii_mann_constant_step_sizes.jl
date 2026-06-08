using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_krasnoselskii_mann_constant_step_sizes(n, gamma; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_krasnoselskii_mann_constant_step_sizes`.

Consider the fixed point problem

```math
\mathrm{Find}\, x:\, x = Ax,
```

where $A$ is a non-expansive operator, that is a $L$-Lipschitz operator with $L=1$.

# Performance metric

This code computes a worst-case guarantee for the **Krasnolselskii-Mann** (KM) method with constant step-size.
That is, it computes the smallest possible $\tau(n)$ such that the guarantee

```math
\frac{1}{4}\|x_n - Ax_n\|^2 \leqslant \tau(n) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of the KM method, and $x_\star$ is some fixed point of $A$
(i.e., $x_\star=Ax_\star$).

# Algorithm
The constant step-size KM method is described by

```math
x_{t+1} = \left(1 - \gamma\right) x_{t} + \gamma Ax_{t}.
```

# Theoretical guarantee
A theoretical **upper** bound is provided by [1, Theorem 4.9]

```math
\tau(n) = \left\{
```
                  \begin{aligned}
                      \frac{1}{n+1}\left(\frac{n}{n+1}\right)^n \frac{1}{4 \gamma (1 - \gamma)}\quad & \text{if } \frac{1}{2}\leqslant \gamma  \leqslant \frac{1}{2}\left(1+\sqrt{\frac{n}{n+1}}\right) \\
                      (\gamma - 1)^{2n} \quad & \text{if } \frac{1}{2}\left(1+\sqrt{\frac{n}{n+1}}\right) <  \gamma \leqslant  1.
                  \end{aligned}
                  \right.

**Reference**:

[[1] F. Lieder (2018).
Projection Based Methods for Conic Linear Programming
Optimal First Order Complexities and Norm Constrained Quasi Newton Methods.
PhD thesis, HHU Dusseldorf.](https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-49971/Dissertation.pdf)

# References
No bibliographic reference was listed in the corresponding Python PEPit example docstring.

# Arguments
- `n`: number of iterations.
- `gamma`: step-size parameter.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(3, 3 / 4; verbose=true)
```
"""
function wc_krasnoselskii_mann_constant_step_sizes(n, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => 1.0)
    A = declare_function!(problem, LipschitzOperator, param)


    xs, _, _ = fixed_point!(A)
    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        x = (1 - gamma) * x + gamma * gradient!(A, x)
    end


    set_performance_metric!(problem, (1 / 2 * (x - gradient!(A, x)))^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if 1 / 2 <= gamma <= 1 / 2 * (1 + sqrt(n / (n + 1)))
        theoretical_tau = 1 / (n + 1) * (n / (n + 1))^n / (4 * gamma * (1 - gamma))
    elseif 1 / 2 * (1 + sqrt(n / (n + 1))) < gamma <= 1
        theoretical_tau = (2 * gamma - 1)^(2 * n)
    else
        error("$(gamma) is not a valid value for the step-size 'gamma'." *
              " 'gamma' must be a number between 1/2 and 1")
    end

    if verbose
        println("*** Example file: worst-case performance of Kranoselskii-Mann iterations ***")
        println("\tPEPit guarantee:\t 1/4||xN - AxN||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t 1/4||xN - AxN||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(3, 3 / 4; verbose=true)
