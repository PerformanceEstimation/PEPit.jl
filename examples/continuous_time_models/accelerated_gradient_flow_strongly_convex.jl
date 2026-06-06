using PEPit, OrderedCollections, Clarabel

function wc_accelerated_gradient_flow_strongly_convex(mu; psd=true, solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu)
    func = declare_function!(problem, StronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)
    xt_dot = set_initial_point!(problem)


    xt_dot_dot = -2 * sqrt(mu) * xt_dot - gt


    if psd
        lyap = ft - fs + 1 / 2 * (sqrt(mu) * (xt - xs) + xt_dot)^2
        lyap_dot = gt * xt_dot + (sqrt(mu) * (xt - xs) + xt_dot) * (sqrt(mu) * xt_dot + xt_dot_dot)
    else
        lyap = ft - fs + mu * 4 / 9 * (xt - xs)^2 + 2 * 2 / 3 * sqrt(mu) * (xt - xs) * xt_dot + 1 / 2 * xt_dot^2
        lyap_dot = gt * xt_dot + mu * 8 / 9 * (xt - xs) * xt_dot + 4 / 3 * sqrt(mu) * (xt_dot^2 + (xt - xs) * xt_dot_dot) + xt_dot_dot * xt_dot
    end


    set_initial_condition!(problem, lyap == 1)


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if psd
        theoretical_tau = -sqrt(mu)
    else
        theoretical_tau = -4 / 3 * sqrt(mu)
    end
    mu == 0 && @warn "Momentum is tuned for strongly convex functions!"


    if verbose
        println("*** Example file: worst-case performance of an accelerated gradient flow ***")
        println("\tPEPit guarantee:\t d/dt V(X_t,t) <= $(round(pepit_tau, digits=6)) V(X_t,t)")
        println("\tTheoretical guarantee:\t d/dt V(X_t) <= $(round(theoretical_tau, digits=6)) V(X_t,t)")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_flow_strongly_convex(0.1; psd=true, solver=Clarabel.Optimizer, verbose=true)
