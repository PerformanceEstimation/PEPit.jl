using PEPit, OrderedCollections, Mosek, MosekTools

function wc_gradient_descent_qg_convex(L, gamma, n; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, ConvexQGFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 1:n
        x = x - gamma * gradient!(func, x)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / 2 * max(1 / (2 * n * L * gamma + 1), L * gamma)


    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
pepit_tau, theoretical_tau = wc_gradient_descent_qg_convex(L, 0.2 / L, 4; verbose=true)
