using PEPit, OrderedCollections, Mosek, MosekTools

function wc_triple_momentum(mu, L, n; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    kappa = L / mu
    rho = (1 - 1 / sqrt(kappa))
    alpha = (1 + rho) / L
    beta = rho^2 / (2 - rho)
    gamma = rho^2 / (1 + rho) / (2 - rho)
    delta = rho^2 / (1 - rho^2)


    x_old = x0
    x_new = x0
    y = x0
    local x
    for _ in 1:n
        x_inter = (1 + beta) * x_new - beta * x_old - alpha * gradient!(func, y)
        y = (1 + gamma) * x_inter - gamma * x_new
        x = (1 + delta) * x_inter - delta * x_new
        x_new, x_old = x_inter, x_new
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = rho^(2 * n) * L / 2 * kappa

    if verbose
        println("*** Example file: worst-case performance of the Triple Momentum Method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0-x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0-x_*||^2")
    end

    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_triple_momentum(0.1, 1.0, 4; solver=Mosek.Optimizer, verbose=true)
