using PEPit, OrderedCollections, Clarabel

function wc_proximal_point(alpha, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    A = declare_function!(problem, MonotoneOperator, OrderedDict())


    xs = stationary_point!(A)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    previous_x = x0
    for _ in 1:n
        previous_x = x
        x, _, _ = proximal_step!(previous_x, A, alpha)
    end


    set_performance_metric!(problem, (x - previous_x)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose, tracetrick=true)


    theoretical_tau = (1 - 1 / n)^(n - 1) / n


    if verbose
        println("*** Example file: worst-case performance of the Proximal Point Method***")
        println("\tPEPit guarantee:\t ||x(n) - x(n-1)||^2 == $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_proximal_point(2.2, 11; solver=Clarabel.Optimizer, verbose=true)
