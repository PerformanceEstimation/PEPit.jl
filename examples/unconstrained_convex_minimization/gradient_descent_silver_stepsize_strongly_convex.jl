using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_silver_stepsize_strongly_convex(L, mu, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_silver_stepsize_strongly_convex`.

Consider the strongly convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$ strongly-convex.

# Performance metric

This code computes a worst-case guarantee for $n$ steps of the **gradient descent** method tuned
according to the silver stepsize schedule.
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
\|x_n - x_\star\|^2 \leqslant \tau(n, L, \mu) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of gradient descent using the silver stepsizes, and
where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$, $L$ and $\mu$, $\tau(n, L, \mu)$ is computed
as the worst-case value of $\|x_n - x_\star\|^2$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

Gradient descent is described by

```math
x_{t+1} = x_t - \gamma_t \nabla f(x_t),
```

where $\gamma_t$ is a step-size of the $t^{th}$ step of the silver step-size schedule described in [1].

# Theoretical guarantee

The theoretical guarantee for the convergence rate of the silver stepsize can be found in [1, Theorem 4.1]:
Let $n^\star = 2^{\lfloor log_\rho(L/(3\mu)) \rfloor}$.

When $n \leq n^\star$, the guarantee is given by

```math
\|x_n - x_\star\|^2 \leqslant e^{-\frac{n^{\log_2(1 + \sqrt{2})}}{L/\mu}} \|x_0-x_\star\|^2,
```

When $n > n^\star$ the guarantee is given by

```math
\|x_n - x_\star\|^2 \leqslant e^{-\frac{n}{n^*} \frac{(n^*)^{\log_2(\rho)}}{L/\mu}} \|x_0-x_\star\|^2
```

# References


[[1] J. M. Altschuler, P. A. Parrilo (2023).
Acceleration by Stepsize Hedging I: Multi-Step Descent and the Silver Stepsize Schedule.
arXiv preprint arXiv:2309.07879.](https://arxiv.org/abs/2309.07879)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_strongly_convex(3.2, 0.1, 8; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.22145, 0.22145)
```
"""
function wc_gradient_descent_silver_stepsize_strongly_convex(L, mu, n; solver=Clarabel.Optimizer, verbose=true)


    if !isinteger(log2(n))
        @warn "Silver step-size strategy is optimally designed when n is a power of 2." *
              " The provided input n is not a power of 2." *
              " We decompose n as sum_k 2^k and recursely use sequences of stepsizes of length 2^k."
    end


    nbits = ndigits(n, base=2)
    n_glue_list = [i for i in 0:(nbits - 1) if (n & (1 << i)) != 0]


    h = Float64[]
    theoretical_tau = 1.0


    psi(t) = (1 + L / mu * t) / (1 + t)


    for n_glue in n_glue_list


        y = [mu / L]
        z = [mu / L]

        a = [psi(y[1])]
        b = [psi(z[1])]

        h_temp = [b[1]]
        for step in 1:n_glue
            z_old = z[step]
            eta = 1 - z_old
            y_new = z_old / (eta + sqrt(1 + eta^2))
            z_new = z_old * (eta + sqrt(1 + eta^2))
            push!(y, y_new)
            push!(z, z_new)
            a_new = psi(y_new)
            b_new = psi(z_new)
            push!(a, a_new)
            push!(b, b_new)
            h_tilde = h_temp[1:end-1]
            h_temp = vcat(h_tilde, [a_new], h_tilde, [b_new])
        end


        h = vcat(h, h_temp)


        theoretical_tau *= ((1 - z[end]) / (1 + z[end]))^2
    end


    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 1:n
        x = x - h[i] / L * gradient!(func, x)
    end


    set_performance_metric!(problem, (x - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if verbose
        println("*** Example file: worst-case performance of gradient descent with silver step-sizes ***")
        println("\tPEPit guarantee:\t ||x_n - x_*||^2 <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||x_n - x_*||^2 <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_strongly_convex(3.2, 0.1, 8; verbose=true)
