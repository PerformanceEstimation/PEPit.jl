using PEPit, OrderedCollections, Clarabel

function wc_conjugate_gradient_qg_convex(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexQGFunction, OrderedDict("L" => L))


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    g0, f0 = oracle!(func, x0)
    span = [g0]
    local gx, fx
    for i in 1:n
        x_old = x_new
        x_new, gx, fx = exact_linesearch_step!(x_new, func, span)
        push!(span, gx)
        push!(span, x_old - x_new)
    end


    set_performance_metric!(problem, fx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * (n + 1))

    if verbose
        println("*** Example file: worst-case performance of conjugate gradient method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_conjugate_gradient_qg_convex(1.0, 12; solver=Clarabel.Optimizer, verbose=true)
