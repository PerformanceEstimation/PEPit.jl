using PEPit, OrderedCollections, Mosek, MosekTools

function wc_robust_momentum(mu, L, lam; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    x1 = set_initial_point!(problem)


    kappa = L / mu
    rho = lam * (1 - 1 / kappa) + (1 - lam) * (1 - 1 / sqrt(kappa))
    alpha = kappa * (1 - rho)^2 * (1 + rho) / L
    beta = kappa * rho^3 / (kappa - 1)
    gamma = rho^3 / ((kappa - 1) * (1 - rho)^2 * (1 + rho))
    l = mu^2 * (kappa - kappa * rho^2 - 1) / (2 * rho * (1 - rho))


    y0 = x1 + gamma * (x1 - x0)
    g0, f0 = oracle!(func, y0)
    x2 = x1 + beta * (x1 - x0) - alpha * g0
    y1 = x2 + gamma * (x2 - x1)
    g1, f1 = oracle!(func, y1)
    x3 = x2 + beta * (x2 - x1) - alpha * g1

    z1 = (x2 - (rho^2) * x1) / (1 - rho^2)
    z2 = (x3 - (rho^2) * x2) / (1 - rho^2)


    q0 = (L - mu) * (f0 - fs - mu / 2 * (y0 - xs)^2) - 1 / 2 * (g0 - mu * (y0 - xs))^2
    q1 = (L - mu) * (f1 - fs - mu / 2 * (y1 - xs)^2) - 1 / 2 * (g1 - mu * (y1 - xs))^2
    initLyapunov = l * (z1 - xs)^2 + q0
    finalLyapunov = l * (z2 - xs)^2 + q1


    set_initial_condition!(problem, initLyapunov <= 1)


    set_performance_metric!(problem, finalLyapunov)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = rho^2


    if verbose
        println("*** Example file: worst-case performance of the Robust Momentum Method ***")
        println("\tPEPit guarantee:\t v(x_(n+1)) <= $(round(pepit_tau, digits=6)) v(x_n)")
        println("\tTheoretical guarantee:\t v(x_(n+1)) <= $(round(theoretical_tau, digits=6)) v(x_n)")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_robust_momentum(0.1, 1.0, 0.2; solver=Mosek.Optimizer, verbose=true)
