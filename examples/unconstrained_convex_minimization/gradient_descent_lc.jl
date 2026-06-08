using PEPit, OrderedCollections, Clarabel


function _fsolve_scalar(fun, x0; xtol=1e-10, maxiter=200)
    x = x0
    for _ in 1:maxiter
        fx = fun(x)
        h = 1e-8 * max(abs(x), 1.0)
        dfx = (fun(x + h) - fun(x - h)) / (2 * h)
        dfx == 0 && break
        dx = fx / dfx
        x -= dx
        abs(dx) < xtol && break
    end
    return x
end

@doc raw"""
    wc_gradient_descent_lc(mug, Lg, typeM, muM, LM, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_gradient_descent_lc`.

Consider the convex minimization problem

```math
g_\star \triangleq \min_x g(Mx),
```

where $g$ is an $L_g$-smooth, `\mu_g`-strongly convex function and $M$ is a general, symmetric or
skew-symmetric matrix with $\mu_M \leqslant \|M\| \leqslant L_M$.

# Performance metric

This code computes a worst-case guarantee for the objective residual

```math
g(Mx_n) - g_\star
```

under the normalization $\|x_0-x_\star\|^2 \leqslant 1$, where $x_\star$ is a
stationary point of the linearly composed objective.

# Algorithm

The method is gradient descent on the composition $x \mapsto g(Mx)$. The
symbolic step is

```math
x_{t+1} = x_t - \gamma M^\ast \nabla g(Mx_t),
```

where the Julia implementation realizes $M^\ast$ as `M.T` for a general linear
operator, as `M` for a symmetric operator, and as `-M` for a skew-symmetric
operator.

# Theoretical guarantee

The example computes its reference value from the scalar equation and closed-form
post-processing implemented below, using the condition ratios
$\kappa_g=\mu_g/L_g$ and $\kappa_M=\mu_M/L_M$.

# References
No bibliographic reference was listed in the corresponding Python PEPit example docstring.

# Arguments
- `mug`: the strong convexity parameter of $g(y)$.
- `Lg`: the smoothness parameter of $g(y)$.
- `typeM`: type of matrix $M$ ("gen", "sym" or "skew").
- `muM`: lower bound on $\|M\|$ (if typeM != "sym", then muM must be set to zero).
- `LM`: upper bound on $\|M\|$.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_gradient_descent_lc(mug, Lg, typeM, muM, LM, 1 / (Lg * LM^2), 3; verbose=true)
```
"""
function wc_gradient_descent_lc(mug, Lg, typeM, muM, LM, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    G = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mug, "L" => Lg))
    if typeM == "gen"
        M = declare_function!(problem, LinearOperator, OrderedDict("L" => LM))
    elseif typeM == "sym"
        M = declare_function!(problem, SymmetricLinearOperator, OrderedDict("mu" => muM, "L" => LM))
    elseif typeM == "skew"
        M = declare_function!(problem, SkewSymmetricLinearOperator, OrderedDict("L" => LM))
    else
        error("The argument 'typeM' must be 'gen', 'sym' or 'skew'. Got $(typeM)")
    end


    x0 = set_initial_point!(problem)


    xs = Point()
    ys = gradient!(M, xs)
    us, fs = oracle!(G, ys)
    if typeM == "gen"
        vs = gradient!(M.T, us)
    elseif typeM == "sym"
        vs = gradient!(M, us)
    elseif typeM == "skew"
        vs = -gradient!(M, us)
    else
        error("The argument 'typeM' must be 'gen', 'sym' or 'skew'. Got $(typeM)")
    end
    add_constraint!(problem, vs^2 == 0)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for _ in 1:n
        y = gradient!(M, x)
        u = gradient!(G, y)
        if typeM == "gen"
            v = gradient!(M.T, u)
        elseif typeM == "sym"
            v = gradient!(M, u)
        elseif typeM == "skew"
            v = -gradient!(M, u)
        else
            error("The argument 'typeM' must be 'gen', 'sym' or 'skew'. Got $(typeM)")
        end
        x = x - gamma * v
    end


    set_performance_metric!(problem, value!(G, gradient!(M, x)) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    L = Lg * LM^2
    kappag = mug / Lg
    kappaM = muM / LM

    fun(z) = (1 - (2 * n + 1) * z) * (1 - z)^(-2 * n - 1) - 1 + kappag

    xroot = _fsolve_scalar(fun, 0.5; xtol=1e-10)
    h0 = xroot / kappag
    t = sqrt(h0 / (L * gamma))

    if t < kappaM
        M_star = kappaM
    elseif t > 1
        M_star = 1.0
    else
        M_star = t
    end

    theoretical_tau = 0.5 * L * max(
        kappag * M_star^2 / (kappag - 1 + (1 - kappag * L * gamma * M_star^2)^(-2 * n)),
        (1 - L * gamma)^(2 * n))


    if verbose
        println("*** Example file: worst-case performance of gradient descent on g(Mx) with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


Lg = 3.0
mug = 0.3
typeM = "gen"
LM = 1.0
muM = 0.1

pepit_tau, theoretical_tau = wc_gradient_descent_lc(mug, Lg, typeM, muM, LM, 1 / (Lg * LM^2), 3; verbose=true)
