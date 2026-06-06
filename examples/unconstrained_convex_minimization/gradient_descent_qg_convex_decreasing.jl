using PEPit, OrderedCollections, Mosek, MosekTools

function wc_gradient_descent_qg_convex_decreasing(L, n; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, ConvexQGFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x = set_initial_point!(problem)
    g, f = oracle!(func, x)


    set_initial_condition!(problem, (x - xs)^2 <= 1)


    u = 1.0
    for i in 1:n

        u = u / 2 + sqrt((u / 2)^2 + 2)
        gamma = 1 / (L * u)
        x = x - gamma * g
        g, f = oracle!(func, x)
    end


    theoretical_tau = L / (2 * u)


    set_performance_metric!(problem, f - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical conjecture:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex_decreasing(1.0, 6; verbose=true)
