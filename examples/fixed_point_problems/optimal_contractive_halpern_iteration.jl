using PEPit, OrderedCollections, Clarabel

function wc_optimal_contractive_halpern_iteration(n, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => 1 / gamma)
    A = declare_function!(problem, LipschitzOperator, param; reuse_gradient=true)


    xs, _, _ = fixed_point!(A)
    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        phi = (gamma^(2 * i + 4) - 1) / (gamma^2 - 1)
        x = 1 / phi * x0 + (1 - 1 / phi) * gradient!(A, x)
    end


    set_performance_metric!(problem, (x - gradient!(A, x))^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)
    theoretical_tau = (1 + 1 / gamma)^2 * ((gamma - 1) / (gamma^(n + 1) - 1))^2

    if verbose
        println("*** Example file: worst-case performance of Optimal Contractive Halpern Iterations ***")
        println("\tPEPit guarantee:\t ||xN - AxN||^2 <= $(round(pepit_tau, digits=7)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||xN - AxN||^2 <= $(round(theoretical_tau, digits=7)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_optimal_contractive_halpern_iteration(10, 1.1; solver=Clarabel.Optimizer, verbose=true)
