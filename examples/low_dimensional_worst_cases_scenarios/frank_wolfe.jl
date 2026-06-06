using PEPit, OrderedCollections, Clarabel

function wc_frank_wolfe(L, D, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func1 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L))
    func2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))

    func = func1 + func2


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    _ = value!(func1, x0)
    _ = value!(func2, x0)


    x = x0
    for i in 0:(n-1)
        g = gradient!(func1, x)
        y, _, _ = linear_optimization_step!(g, func2)
        lam = 2 / (i + 2)
        x = (1 - lam) * x + lam * y
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=12)


    theoretical_tau = 2 * L * D^2 / (n + 2)


    if verbose
        println("*** Example file: worst-case performance of the Conditional Gradient (Frank-Wolfe) in function value ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* == $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_frank_wolfe(1.0, 1.0, 10; solver=Clarabel.Optimizer, verbose=true)
