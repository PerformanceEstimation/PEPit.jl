using PEPit
using OrderedCollections

function wc_inexact_gradient_exact_line_search(L, mu, epsilon, n; verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param; reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x = set_initial_point!(problem)
    set_initial_condition!(problem, value!(func, x) - fs <= 1)


    fx = value!(func, x)


    for i in 1:n
        _, dx, _ = inexact_gradient_step!(x, func, 0.0, epsilon; notion="relative")
        x, gx, fx = exact_linesearch_step!(x, func, [dx])
    end


    set_performance_metric!(problem, fx - fs)

    pepit_tau = solve!(problem; verbose=verbose)

    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    theoretical_tau = ((Leps - meps) / (Leps + meps))^(2 * n)

    if verbose
        println("*** Example file: worst-case performance of inexact gradient descent with exact linesearch ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_inexact_gradient_exact_line_search(1.0, 0.1, 0.1, 2; verbose=true)
