using PEPit, OrderedCollections, Clarabel

function wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, n, alpha=0; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu1, "L" => L1))
    f2 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu2, "L" => L2))
    F = f1 - f2


    xs = stationary_point!(F)
    Fs = value!(F, xs)


    x0 = set_initial_point!(problem)

    x = x0
    g1x = gradient!(f1, x0)
    g2x = gradient!(f2, x0)


    set_initial_condition!(problem, value!(f1, x) - value!(f2, x) - Fs <= 1)

    for i in 1:n
        set_performance_metric!(problem, (g1x - g2x)^2)
        add_constraint!(problem, Fs <= value!(f1, x) - value!(f2, x) - 1 / 2 / (L1 - mu2) * (g1x - g2x)^2)
        y, _, _ = shifted_optimization_step!(g2x, f1)
        x = (1 + alpha) * y - alpha * x
        g1x, f1x = oracle!(f1, x)
        g2x, f1x = oracle!(f2, x)
    end

    set_performance_metric!(problem, (g1x - g2x)^2)
    add_constraint!(problem, Fs <= value!(f1, x) - value!(f2, x) - 1 / (2 * (L1 - mu2)) * (g1x - g2x)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if alpha == 0
        Delta = 1
        A = 2 * (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1))
        B = L1 + L2 + mu1 * (L1 / L2 - 3) * (L1 >= L2) + mu2 * (L2 / L1 - 3) * (L2 > L1)
        C = (L1 * L2 - mu1 * L2 * (L1 >= L2) - mu2 * L1 * (L2 > L1)) / (L1 - mu2)
        theoretical_tau = A * Delta / (B * n + C)
    else
        theoretical_tau = nothing
    end

    if verbose
        println("*** Example file: worst-case performance of DCA ***")
        println("\tPEPit guarantee:\t min_i ||f'(x_i)||^2 <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t min_i ||f'(x_i)||^2 <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


L1, L2, mu1, mu2 = 2.0, 5.0, 0.15, 0.1
pepit_tau, theoretical_tau = wc_difference_of_convex_algorithm(mu1, mu2, L1, L2, 5, 0; solver=Clarabel.Optimizer, verbose=true)
