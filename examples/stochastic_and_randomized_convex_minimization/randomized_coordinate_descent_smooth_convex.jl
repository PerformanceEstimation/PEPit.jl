using PEPit, OrderedCollections, Clarabel

function wc_randomized_coordinate_descent_smooth_convex(L, gamma, d, t; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    partition = declare_block_partition!(problem, d)


    func = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt_minus_1 = set_initial_point!(problem)


    phi(k, x) = (k * gamma * L / d + 1) * (value!(func, x) - fs) + L / 2 * (x - xs)^2


    set_initial_condition!(problem, phi(t - 1, xt_minus_1) <= 1)


    gt_minus_1 = gradient!(func, xt_minus_1)
    xt_list = [xt_minus_1 - gamma * get_block(partition, gt_minus_1, i) for i in 1:d]


    phi_t = sum(phi(t, xt) for xt in xt_list) / d


    set_performance_metric!(problem, phi_t)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 1.0


    if verbose
        println("*** Example file: worst-case performance of randomized  coordinate gradient descent ***")
        println("\tPEPit guarantee:\t E[phi(t, x_t)] <= $(round(pepit_tau, digits=6)) phi(t-1, x_(t-1))")
        println("\tTheoretical guarantee:\t E[phi(t, x_t)] <= $(round(theoretical_tau, digits=6)) phi(t-1, x_(t-1))")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_convex(L, 1 / L, 2, 4; solver=Clarabel.Optimizer, verbose=true)
