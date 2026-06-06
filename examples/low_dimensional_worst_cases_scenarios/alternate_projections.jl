using PEPit, OrderedCollections, Clarabel

function wc_alternate_projections(n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    ind_Q1 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    ind_Q2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict())
    func = ind_Q1 + ind_Q2


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    x = x0
    for _ in 1:n
        y, _, _ = proximal_step!(x, ind_Q1, 1)
        x, _, _ = proximal_step!(y, ind_Q2, 1)
    end


    proj1_x, _, _ = proximal_step!(x, ind_Q1, 1)
    proj2_x = x
    set_performance_metric!(problem, (proj2_x - proj1_x)^2)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=1)
    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case performance of the alternate projection method ***")
        println("\tPEPit guarantee:\t ||Proj_Q1 (xn) - Proj_Q2 (xn)||^2 == $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_alternate_projections(10; solver=Clarabel.Optimizer, verbose=true)
