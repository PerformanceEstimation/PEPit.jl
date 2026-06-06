using PEPit, OrderedCollections, Clarabel

function wc_subgradient_method(M, n, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    func = declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    gx, fx = oracle!(func, x)

    for _ in 1:n
        set_performance_metric!(problem, fx - fs)
        x = x - gamma * gx
        gx, fx = oracle!(func, x)
    end


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = M / sqrt(n + 1)

    if verbose
        println("*** Example file: worst-case performance of subgradient method ***")
        println("\tPEPit guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||")
        println("\tTheoretical guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||")
    end

    return pepit_tau, theoretical_tau
end


M = 2
n = 6
gamma = 1 / (M * sqrt(n + 1))
pepit_tau, theoretical_tau = wc_subgradient_method(M, n, gamma; verbose=true)
