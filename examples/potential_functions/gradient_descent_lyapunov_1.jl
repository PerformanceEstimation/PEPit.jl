using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_lyapunov_1(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_lyapunov_1`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and convex.

# Performance metric

This code verifies a worst-case guarantee for **gradient descent** with fixed step-size $\gamma$.
That is, it verifies that the Lyapunov (or potential/energy) function

```math
V_n \triangleq n (f(x_n) - f_\star) + \frac{L}{2} \|x_n - x_\star\|^2
```

is decreasing along all trajectories and all smooth convex function $f$ (i.e., in the worst-case):

```math
V_{n+1} \leqslant V_n,
```

where $x_{n+1}$ is obtained from a gradient step from $x_{n}$
with fixed step-size $\gamma=\frac{1}{L}$.

# Algorithm
Onte iteration of gradient descent is described by

```math
x_{n+1} = x_n - \gamma \nabla f(x_n),
```

where $\gamma$ is a step-size.

# Theoretical guarantee
The theoretical guarantee can be found in e.g., [1, Theorem 3.3]:

```math
V_{n+1} - V_n \leqslant 0,
```

when $\gamma=\frac{1}{L}$.

# References
The detailed potential function can found in [1] and the SDP approach can be found in [2].

[[1] N. Bansal, A. Gupta (2019).
Potential-function proofs for gradient methods.
Theory of Computing, 15(1), 1-32.](https://arxiv.org/pdf/1712.04581.pdf)

[[2] A. Taylor, F. Bach (2019).
Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions.
Conference on Learning Theory (COLT).](https://arxiv.org/pdf/1902.00947.pdf)

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
pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_1(1.0, 1.0, 10; verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.0, 0.0)
```
"""
function wc_gradient_descent_lyapunov_1(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xn = set_initial_point!(problem)
    gn, fn = oracle!(func, xn)


    xnp1 = xn - gamma * gn
    gnp1, fnp1 = oracle!(func, xnp1)


    init_lyapunov = n * (fn - fs) + L / 2 * (xn - xs)^2
    final_lyapunov = (n + 1) * (fnp1 - fs) + L / 2 * (xnp1 - xs)^2


    set_performance_metric!(problem, final_lyapunov - init_lyapunov)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = gamma == 1 / L ? 0.0 : nothing

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-size for a given Lyapunov function ***")
        println("\tPEPit guarantee:\tV_(n+1) - V_(n) <= $(round(pepit_tau, digits=6))")
        if gamma == 1 / L
            println("\tTheoretical guarantee:\tV_(n+1) - V_(n) <= $(round(theoretical_tau, digits=6))")
        end
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_1(1.0, 1.0, 10; verbose=true)
