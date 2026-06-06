using PEPit, OrderedCollections, Clarabel

function wc_gradient_flow_strongly_convex(mu; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu)
    func = declare_function!(problem, StronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xt = set_initial_point!(problem)
    gt, ft = oracle!(func, xt)


    xt_dot = -gt


    lyap = ft - fs
    lyap_dot = gt * xt_dot


    set_initial_condition!(problem, lyap == 1)


    set_performance_metric!(problem, lyap_dot)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = -2 * mu
    mu == 0 && @warn "Momentum is tuned for strongly convex functions!"


    if verbose
        println("*** Example file: worst-case performance of the gradient flow ***")
        println("\tPEPit guarantee:\t d/dt[f(X_t)-f_*] <= $(round(pepit_tau, digits=6)) (f(X_t) - f(x_*))")
        println("\tTheoretical guarantee:\t d/dt[f(X_t)-f_*] <= $(round(theoretical_tau, digits=6)) (f(X_t) - f(x_*))")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_gradient_flow_strongly_convex(0.1; solver=Clarabel.Optimizer, verbose=true)
