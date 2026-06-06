using PEPit, OrderedCollections, Clarabel

function wc_proximal_point(gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    fx = fs
    for _ in 1:n
        x, _, fx = proximal_step!(x, func, gamma)
    end


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 1 / (4 * gamma * n)


    if verbose
        println("*** Example file: worst-case performance of proximal point method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_proximal_point(3, 4; solver=Clarabel.Optimizer, verbose=true)
