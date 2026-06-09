using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_gradient_descent_quadratic_lojasiewicz_naive(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_quadratic_lojasiewicz_naive`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and satisfies a quadratic Lojasiewicz inequality:

```math
f(x)-f_\star \leqslant \frac{1}{2\mu}\|\nabla f(x) \|^2,
```

details can be found in [1,2,3].

# Performance metric

This code computes a worst-case guarantee for **gradient descent** with fixed step-size $\gamma$.
That is, it computes the smallest possible $\tau(n, L, \gamma)$ such that the guarantee

```math
f(x_n)-f_\star \leqslant \tau(n, L, \mu, \gamma) (f(x_0) - f(x_\star))
```

is valid, where $x_n$ is the n-th iterates obtained with the gradient method with fixed step-size.

# Algorithm

Gradient descent is described as follows, for $t \in \{ 0, \dots, n-1\}$,

```math
x_{t+1} = x_t - \gamma \nabla f(x_t),
```

where $\gamma$ is a step-size and.

# Theoretical guarantee
We compare with the guarantees from [4, Theorem 3].

# References

    [[1] S. Lojasiewicz (1963).
    Une propriete topologique des sous-ensembles analytiques reels.
    Les equations aux derivees partielles, 117 (1963), 87-89.](https://aif.centre-mersenne.org/item/10.5802/aif.1384.pdf)

    [[2] B. Polyak (1963).
    Gradient methods for the minimisation of functionals
    USSR Computational Mathematics and Mathematical Physics 3(4), 864-878.](https://www.sciencedirect.com/science/article/abs/pii/0041555363903823)

    [[3] J. Bolte, A. Daniilidis, and A. Lewis (2007).
    The ojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems.
    SIAM Journal on Optimization 17, 1205-1223.](https://bolte.perso.math.cnrs.fr/Loja.pdf)

    [[4] H. Abbaszadehpeivasti, E. de Klerk, M. Zamani (2023).
    Conditions for linear convergence of the gradient method for non-convex optimization.
    Optimization Letters.](https://arxiv.org/pdf/2204.00647)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent_quadratic_lojasiewicz_naive(1.0, 0.2, 1.0, 1; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.727273, 0.727273)
```
"""
function wc_gradient_descent_quadratic_lojasiewicz_naive(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L, "mu" => mu)
    func = declare_function!(problem, SmoothQuadraticLojasiewiczFunctionCheap, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, value!(func, x0) - fs <= 1)


    x = x0
    for i in 1:n
        g = gradient!(func, x)
        x = x - gamma * g
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    m, mp = -L, mu
    if 0 <= gamma <= 1 / L
        theoretical_tau = (mp * (1 - L * gamma) + sqrt(
            (L - m) * (m - mp) * (2 - L * gamma) * mp * gamma + (L - m)^2)^2 / (L - m + mp)^2)
    elseif 1 / L <= gamma <= 3 / (m + L + sqrt(m^2 - L * m + L^2))
        theoretical_tau = ((L * gamma - 2) * (m * gamma - 2) * mp * gamma) / ((L + m - mp) * gamma - 2) + 1
    elseif 3 / (m + L + sqrt(m^2 - m * L + L^2)) <= gamma <= 2 / L
        theoretical_tau = (L * gamma - 1)^2 / ((L * gamma - 1)^2 + mp * gamma * (2 - L * gamma))
    else
        theoretical_tau = nothing
    end

    theoretical_tau = theoretical_tau^n

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-size ***")
        println("*** \t (smooth problem satisfying a Lojasiewicz inequality; cheap naive version) ***")
        println("\tPEPit guarantee:\t f(x_1) - f(x_*) <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_1) - f(x_*) <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_descent_quadratic_lojasiewicz_naive(1.0, 0.2, 1.0, 1; solver=Clarabel.Optimizer, verbose=true)
