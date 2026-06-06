using PEPit, OrderedCollections, Clarabel

function wc_gradient_descent(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, SmoothFunction, OrderedDict("L" => L))


    x0 = set_initial_point!(problem)
    g0, f0 = oracle!(func, x0)


    x = x0
    gx, fx = g0, f0


    set_performance_metric!(problem, gx^2)

    for i in 1:n
        x = x - gamma * gx

        gx, fx = oracle!(func, x)
        set_performance_metric!(problem, gx^2)
    end


    set_initial_condition!(problem, f0 - fx <= 1)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, logdetiters=2)


    theoretical_tau = 4 / 3 * L / n


    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-size ***")
        println("\tPEPit guarantee:\t min_i ||f'(x_i)||^2 == $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t min_i ||f'(x_i)||^2 <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
gamma = 1 / L
pepit_tau, theoretical_tau = wc_gradient_descent(L, gamma, 5; solver=Clarabel.Optimizer, verbose=true)
