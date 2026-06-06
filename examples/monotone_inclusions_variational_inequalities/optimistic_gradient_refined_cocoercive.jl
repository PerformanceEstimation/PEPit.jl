using PEPit, OrderedCollections, Clarabel

function wc_optimistic_gradient_refined_cocoercive(n, gamma, beta; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    ind_C = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    F = declare_function!(problem, CocoerciveStronglyMonotoneOperatorExpensive, OrderedDict("mu" => 0, "beta" => beta))

    total_problem = F + ind_C


    xs = stationary_point!(total_problem)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x, _, _ = proximal_step!(x0, ind_C, gamma)
    xtilde = x
    V = gradient!(F, xtilde)
    previous_xtilde = xtilde
    for _ in 1:n
        previous_xtilde = xtilde
        xtilde, _, _ = proximal_step!(x - gamma * V, ind_C, gamma)
        previous_V = V
        V = gradient!(F, xtilde)
        x = xtilde + gamma * (previous_V - V)
    end


    set_performance_metric!(problem, (xtilde - previous_xtilde)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case performance of the Optimistic Gradient Method***")
        println("\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_optimistic_gradient_refined_cocoercive(1, 1 / 4, 1 / 4; verbose=true)
