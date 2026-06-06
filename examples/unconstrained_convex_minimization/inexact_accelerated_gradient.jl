using PEPit, OrderedCollections, Clarabel

function wc_inexact_accelerated_gradient(L, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    y = x0
    for i in 0:(n - 1)
        x_old = x_new
        x_new, dy, fy = inexact_gradient_step!(y, func, 1 / L, epsilon; notion="relative")
        y = x_new + i / (i + 3) * (x_new - x_old)
    end
    _, fx = oracle!(func, x_new)


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 * L / (n^2 + 5 * n + 6)

    if verbose
        println("*** Example file: worst-case performance of inexact accelerated gradient method ***")
        println("\tPEPit guarantee:\t\t\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee for epsilon = 0 :\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_inexact_accelerated_gradient(1.0, 0.1, 5; solver=Clarabel.Optimizer, verbose=true)
