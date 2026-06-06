using PEPit, OrderedCollections, Mosek, MosekTools

function wc_epsilon_subgradient_method(M, n, gamma, eps, R; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= R^2)


    x = x0
    for _ in 1:n
        x, gx, fx, epsilon = epsilon_subgradient_step!(x, func, gamma)
        set_performance_metric!(problem, fx - fs)
        add_constraint!(problem, epsilon <= eps)
        add_constraint!(problem, gx^2 <= M^2)
    end


    gx, fx = oracle!(func, x)
    add_constraint!(problem, gx^2 <= M^2)
    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (R^2 + 2 * (n + 1) * gamma * eps + (n + 1) * gamma^2 * M^2) / (2 * (n + 1) * gamma)

    if verbose
        println("*** Example file: worst-case performance of the epsilon-subgradient method ***")
        println("\tPEPit guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t min_(0 <= t <= n) f(x_i) - f_* <= $(round(theoretical_tau, digits=6))")
    end


    return pepit_tau, theoretical_tau
end


M, n, eps, R = 2, 6, 0.1, 1
gamma = 1 / sqrt(n + 1)
pepit_tau, theoretical_tau = wc_epsilon_subgradient_method(M, n, gamma, eps, R; verbose=true)
