using PEPit, OrderedCollections, Clarabel

function wc_polyak_steps_in_distance_to_optimum(L, mu, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    g0, f0 = oracle!(func, x0)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x1 = x0 - gamma * g0
    _, _ = oracle!(func, x1)


    add_constraint!(problem, gamma * g0^2 == 2 * (f0 - fs))


    set_performance_metric!(problem, (x1 - xs)^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 / L <= gamma <= 1 / mu) ?
        (gamma * L - 1) * (1 - gamma * mu) / (gamma * (L + mu) - 1) :
        0.0

    if verbose
        println("*** Example file: worst-case performance of Polyak steps ***")
        println("\tPEPit guarantee:\t ||x_1 - x_*||^2 <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2 ")
        println("\tTheoretical guarantee:\t ||x_1 - x_*||^2 <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_polyak_steps_in_distance_to_optimum(1.0, 0.1, 2 / (1.0 + 0.1); solver=Clarabel.Optimizer, verbose=true)
