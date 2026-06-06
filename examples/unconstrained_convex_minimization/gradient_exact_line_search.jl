using PEPit
using OrderedCollections

function wc_gradient_exact_line_search(L, mu, n; verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param; reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x = set_initial_point!(problem)
    gx, f0 = oracle!(func, x)


    set_initial_condition!(problem, f0 - fs <= 1)


    fx = f0
    for i in 1:n
        x, gx, fx = exact_linesearch_step!(x, func, [gx])
    end


    set_performance_metric!(problem, fx - fs)

    pepit_tau = solve!(problem; verbose=verbose)
    theoretical_tau = ((L - mu) / (L + mu))^(2 * n)

    if verbose
        println("*** Example file: worst-case performance of gradient descent with exact linesearch (ELS) ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_exact_line_search(1.0, 0.1, 2; verbose=true)
