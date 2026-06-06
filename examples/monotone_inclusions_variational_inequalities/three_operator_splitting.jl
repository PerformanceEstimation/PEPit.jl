using PEPit, OrderedCollections, Clarabel

function wc_three_operator_splitting(L, mu, beta, alpha, theta; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, MonotoneOperator, OrderedDict())
    B = declare_function!(problem, CocoerciveOperator, OrderedDict("beta" => beta))
    C = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


    w0 = set_initial_point!(problem)
    w1 = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w1)^2 <= 1)


    x0, _, _ = proximal_step!(w0, B, alpha)
    y0, _, _ = proximal_step!(2 * x0 - w0 - alpha * gradient!(C, x0), A, alpha)
    z0 = w0 - theta * (x0 - y0)


    x1, _, _ = proximal_step!(w1, B, alpha)
    y1, _, _ = proximal_step!(2 * x1 - w1 - alpha * gradient!(C, x1), A, alpha)
    z1 = w1 - theta * (x1 - y1)


    set_performance_metric!(problem, (z0 - z1)^2)


    pepit_verbose = verbose != false
    pepit_tau = solve!(problem; solver=solver, verbose=pepit_verbose)


    theoretical_tau = nothing


    if verbose
        println("*** Example file: worst-case contraction factor of the Three Operator Splitting ***")
        println("\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(pepit_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_three_operator_splitting(1.0, 0.1, 1.0, 0.9, 1.3; verbose=true)
