using PEPit, OrderedCollections, Clarabel

function wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, d; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    partition = declare_block_partition!(problem, d)


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    g0 = gradient!(func, x0)
    x1_list = [x0 - gamma * get_block(partition, g0, i) for i in 1:d]


    set_performance_metric!(problem, sum((x1 - xs)^2 for x1 in x1_list) / d)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max(((mu * gamma - 1)^2 + d - 1) / d, ((L * gamma - 1)^2 + d - 1) / d)


    if verbose
        println("*** Example file: worst-case performance of randomized coordinate gradient descent ***")
        println("\tPEPit guarantee:\t E[||x_(t+1) - x_*||^2] <= $(round(pepit_tau, digits=6)) ||x_t - x_*||^2")
        println("\tTheoretical guarantee:\t E[||x_(t+1) - x_*||^2] <= $(round(theoretical_tau, digits=6)) ||x_t - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
mu = 0.1
gamma = 2 / (mu + L)
pepit_tau, theoretical_tau = wc_randomized_coordinate_descent_smooth_strongly_convex(L, mu, gamma, 2; solver=Clarabel.Optimizer, verbose=true)
