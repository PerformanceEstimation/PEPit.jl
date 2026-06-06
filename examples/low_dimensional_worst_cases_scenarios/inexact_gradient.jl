using PEPit, OrderedCollections, Clarabel

function wc_inexact_gradient(L, mu, epsilon, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, value!(func, x0) - fs <= 1)


    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    gamma = 2 / (Leps + meps)

    x = x0
    local dx, fx
    for i in 1:n
        x, dx, fx = inexact_gradient_step!(x, func, gamma, epsilon; notion="relative")
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=10)


    theoretical_tau = ((Leps - meps) / (Leps + meps))^(2 * n)


    if verbose
        println("*** Example file: worst-case performance of inexact gradient ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* == $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_inexact_gradient(1.0, 0.1, 0.1, 6; solver=Clarabel.Optimizer, verbose=true)
