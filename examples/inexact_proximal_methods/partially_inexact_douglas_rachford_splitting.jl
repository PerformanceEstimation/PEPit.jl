using PEPit, OrderedCollections, Clarabel


function wc_partially_inexact_douglas_rachford_splitting(mu, L, n, gamma, sigma; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    g = declare_function!(problem, ConvexFunction, OrderedDict())


    F = f + g


    xs = stationary_point!(F)


    zs = xs + gamma * gradient!(f, xs)


    z0 = set_initial_point!(problem)


    set_initial_condition!(problem, (z0 - zs)^2 <= 1)


    z = z0
    for _ in 1:n

        x, dfx, _, _, _, _, eps_var = inexact_proximal_step!(z, f, gamma; opt="PD_gapII")


        y, _, _ = proximal_step!(x - gamma * dfx, g, gamma)


        add_constraint!(f, eps_var <= 1 / 2 * (sigma * (y - z + gamma * dfx))^2)


        z = z + (y - x)
    end


    set_performance_metric!(problem, (z - zs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max(((1 - sigma + gamma * mu * sigma) / (1 - sigma + gamma * mu))^2,
                          ((sigma + (1 - sigma) * gamma * L) / (1 + (1 - sigma) * gamma * L))^2)^n


    if verbose

        println("*** Example file: worst-case performance of the partially inexact Douglas Rachford splitting ***")
        println("\tPEPit guarantee:\t ||z_n - z_*||^2 <= $(round(pepit_tau, digits=6)) ||z_0 - z_*||^2")
        println("\tTheoretical guarantee:\t ||z_n - z_*||^2 <= $(round(theoretical_tau, digits=6)) ||z_0 - z_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_partially_inexact_douglas_rachford_splitting(0.1, 5, 5, 1.4, 0.2; verbose=true)
