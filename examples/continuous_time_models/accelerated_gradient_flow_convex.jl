using PEPit, OrderedCollections, Clarabel

function wc_accelerated_gradient_flow_convex(t; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)
    xt_dot = set_initial_point!(problem)


    xt_dot_dot = -3 / t * xt_dot - gt


    lyap = t^2 * (ft - fs) + 2 * ((xt - xs) + t / 2 * xt_dot)^2
    lyap_dot = 2 * t * (ft - fs) + t^2 * xt_dot * gt + 4 * ((xt - xs) + t / 2 * xt_dot) * (3 / 2 * xt_dot + t / 2 * xt_dot_dot)


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 0.0


    if verbose
        println("*** Example file: worst-case performance of an accelerated gradient flow ***")
        println("\tPEPit guarantee:\t d/dt V(X_t,t) <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t d/dt V(X_t) <= $(round(theoretical_tau, digits=6))")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_convex(3.4; solver=Clarabel.Optimizer, verbose=true)
