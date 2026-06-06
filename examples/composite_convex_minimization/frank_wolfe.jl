using PEPit
using OrderedCollections

function wc_frank_wolfe(L, D, R, center, n; verbose::Bool=true)

    problem = PEP()


    f1 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)
    f2 = declare_function!(problem, ConvexIndicatorFunction,
        OrderedDict("D" => D, "R" => R, "center" => center); reuse_gradient=false)


    F = f1 + f2


    xs = stationary_point!(F)
    fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    _ = value!(f1, x0)
    _ = value!(f2, x0)


    x = x0
    for t in 0:(n-1)
        g = gradient!(f1, x)
        y, _, _ = linear_optimization_step!(g, f2)
        λ = 2 / (t + 2)
        x = (1 - λ) * x + λ * y
    end


    set_performance_metric!(problem, value!(F, x) - fs)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = 2 * L * D^2 / (n + 2)

    if verbose
        println("*** Example file: worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(PEPit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end

    return PEPit_tau, theoretical_tau


end

PEPit_tau, theoretical_tau = wc_frank_wolfe(1.0, 1.0, Inf, nothing, 10; verbose=true)
