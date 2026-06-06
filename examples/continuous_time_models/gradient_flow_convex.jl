using PEPit, OrderedCollections, Clarabel

function wc_gradient_flow_convex(t; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)


    xt_dot = -gt


    lyap_dot = (ft - fs) + t * gt * xt_dot + (xt - xs) * xt_dot


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 0.0


    if verbose
        println("*** Example file: worst-case performance of the gradient flow ***")
        println("\tPEPit guarantee:\t d/dt V(X_t) <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t d/dt V(X_t) <= $(round(theoretical_tau, digits=6))")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_gradient_flow_convex(2.5; solver=Clarabel.Optimizer, verbose=true)
