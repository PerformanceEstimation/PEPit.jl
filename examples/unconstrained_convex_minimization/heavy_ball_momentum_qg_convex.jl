using PEPit, OrderedCollections, Clarabel

function wc_heavy_ball_momentum_qg_convex(L, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, ConvexQGFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    x_old = x0
    for t in 0:(n - 1)
        x_next = x_new - 1 / (L * (t + 2)) * gradient!(func, x_new) + t / (t + 2) * (x_new - x_old)
        x_old = x_new
        x_new = x_next
    end


    set_performance_metric!(problem, value!(func, x_new) - fs)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)
    theoretical_tau = L / (2 * (n + 1))

    if verbose
        println("*** Example file: worst-case performance of the Heavy-Ball method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_heavy_ball_momentum_qg_convex(1, 5; verbose=true)
