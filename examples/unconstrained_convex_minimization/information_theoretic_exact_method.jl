using PEPit, OrderedCollections, Clarabel

function wc_information_theoretic(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)


    z0 = set_initial_point!(problem)


    set_initial_condition!(problem, (z0 - xs)^2 <= 1)


    A_new = 0.0
    q = mu / L

    x = z0
    z = z0

    for i in 1:n
        A_old = A_new
        A_new = ((1 + q) * A_old + 2 * (1 + sqrt((1 + A_old) * (1 + q * A_old)))) / (1 - q)^2
        beta = A_old / (1 - q) / A_new
        delta = 1 / 2 * ((1 - q)^2 * A_new - (1 + q) * A_old) / (1 + q + q * A_old)

        y = (1 - beta) * z + beta * x
        x = y - 1 / L * gradient!(func, y)
        z = (1 - q * delta) * z + q * delta * y - delta / L * gradient!(func, y)
    end


    set_performance_metric!(problem, (z - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 1 / (1 + q * A_new)


    if verbose
        println("*** Example file: worst-case performance of the information theoretic exact method ***")
        println("\tPEP-it guarantee:\t ||z_n - x_* ||^2 <= $(round(pepit_tau, digits=6)) ||z_0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||z_n - x_* ||^2 <= $(round(theoretical_tau, digits=6)) ||z_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_information_theoretic(0.001, 1.0, 15; solver=Clarabel.Optimizer, verbose=true)
