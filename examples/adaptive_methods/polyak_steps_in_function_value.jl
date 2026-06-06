using PEPit, Clarabel
using OrderedCollections

function wc_polyak_steps_in_function_value(L, mu, gamma; verbose=true)
    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu); reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    g0, f0 = oracle!(func, x0)


    set_initial_condition!(problem, f0 - fs <= 1)
    add_constraint!(problem, g0^2 == 2 * L * (2 - L * gamma) * (f0 - fs))


    x1 = x0 - gamma * g0
    g1, f1 = oracle!(func, x1)


    set_performance_metric!(problem, f1 - fs)

    pepit_tau = solve!(problem; solver = Clarabel.Optimizer, verbose=verbose)

    theoretical_tau = (1 / L <= gamma <= (2 * L - mu) / L^2) ?
        (gamma * L - 1) * (L * gamma * (3 - gamma * (L + mu)) - 1) :
        0.0

    if verbose
        println("*** Example file: worst-case performance of Polyak steps ***")
        println("\tPEPit guarantee:\t f(x_1) - f_* <= $(round(pepit_tau, digits=6)) (f(x_0) - f_*) ")
        println("\tTheoretical guarantee:\t f(x_1) - f_* <= $(round(theoretical_tau, digits=6)) (f(x_0) - f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_polyak_steps_in_function_value(1.0, 0.1, 2 / (1 + 0.1); verbose=true)
