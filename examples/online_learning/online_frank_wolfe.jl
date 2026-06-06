using PEPit, OrderedCollections, Clarabel

function wc_online_frank_wolfe(M::Real, D::Real, n::Int; solver = Clarabel.Optimizer, verbose = true)


    problem = PEP()


    eta = D / (2 * M) * (3 / n)^(3 / 4)
    sigma = min(1, sqrt(3 / n))


    fis = [declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M)) for _ in 1:n]


    h = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))


    x_ref = set_initial_point!(problem)
    x_ref, _, _ = proximal_step!(x_ref, h, 1)


    x1 = set_initial_point!(problem)
    x1, _, _ = proximal_step!(x1, h, 1)


    x = x1
    acc_g = 0 * x_ref
    regret = 0 * x_ref^2

    for i in 1:n
        g, f = oracle!(fis[i], x)
        regret = regret + f - value!(fis[i], x_ref)
        acc_g = acc_g + g
        dir_t = (x - x1) + eta * acc_g
        v, _, _ = linear_optimization_step!(dir_t, h)
        x = (1 - sigma) * x + sigma * v
    end


    set_performance_metric!(problem, regret)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 4 / 3^(3 / 4) * M * D * n^(3 / 4)


    if verbose != -1
        println("*** Example file: worst-case regret of online Frank-Wolfe ***")
        println("\tPEPit guarantee:\t R_n <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t R_n <= $(round(theoretical_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


M, D, n = 1.0, 0.5, 2

pepit_tau, theoretical_tau = wc_online_frank_wolfe(M, D, n; verbose=true)
