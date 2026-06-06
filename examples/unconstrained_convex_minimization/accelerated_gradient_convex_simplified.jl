using PEPit, OrderedCollections, Clarabel

function wc_accelerated_gradient_convex_simplified(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    y = x0
    for i in 1:n
        ipy = i - 1
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        y = x_new + ipy / (ipy + 3) * (x_new - x_old)
    end


    set_performance_metric!(problem, value!(func, x_new) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 * L / (n^2 + 5 * n + 6)
    mu != 0 && @warn "Momentum is tuned for non-strongly convex functions."


    if verbose
        println("*** Example file: worst-case performance of accelerated gradient method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_convex_simplified(0, 1, 1; solver=Clarabel.Optimizer, verbose=true)
