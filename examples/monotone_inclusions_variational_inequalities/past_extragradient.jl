using PEPit, OrderedCollections, Clarabel

function wc_past_extragradient(n, gamma, L; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    ind_C = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    F = declare_function!(problem, LipschitzStronglyMonotoneOperatorCheap, OrderedDict("mu" => 0, "L" => L))

    total_problem = F + ind_C


    xs = stationary_point!(total_problem)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x, _, _ = proximal_step!(x0, ind_C, gamma)
    xtilde = x
    V = gradient!(F, xtilde)
    previous_x = x
    for _ in 1:n
        xtilde, _, _ = proximal_step!(x - gamma * V, ind_C, gamma)
        V = gradient!(F, xtilde)
        previous_x = x
        x, _, _ = proximal_step!(x - gamma * V, ind_C, gamma)
    end


    set_performance_metric!(problem, (x - previous_x)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case performance of the Past Extragradient Method***")
        println("\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_past_extragradient(5, 1 / 4, 1; verbose=true)
