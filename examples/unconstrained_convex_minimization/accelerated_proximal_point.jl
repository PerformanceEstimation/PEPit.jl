using PEPit, OrderedCollections, Mosek, MosekTools

function wc_accelerated_proximal_point(A0, gammas, n; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, value!(func, x0) - fs + A0 / 2 * (x0 - xs)^2 <= 1)


    x, v = x0, x0
    A = A0
    for i in 1:n
        alpha = (sqrt((A * gammas[i])^2 + 4 * A * gammas[i]) - A * gammas[i]) / 2
        y = (1 - alpha) * x + alpha * v
        x, _, _ = proximal_step!(y, func, gammas[i])
        v = v + 1 / alpha * (x - y)
        A = (1 - alpha) * A
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    accumulation = 0.0
    for i in 1:n
        accumulation += sqrt(gammas[i])
    end
    theoretical_tau = 4 / A0 / accumulation^2


    if verbose
        println("*** Example file: worst-case performance of fast proximal point method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_proximal_point(5, [(i + 1) / 1.1 for i in 0:2], 3; solver=Mosek.Optimizer, verbose=true)
