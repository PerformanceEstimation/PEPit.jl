using PEPit, OrderedCollections, Clarabel

function wc_krasnoselskii_mann_increasing_step_sizes(n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => 1.0)
    A = declare_function!(problem, LipschitzOperator, param)


    xs, _, _ = fixed_point!(A)
    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        x = 1 / (i + 2) * x + (1 - 1 / (i + 2)) * gradient!(A, x)
    end


    set_performance_metric!(problem, (1 / 2 * (x - gradient!(A, x)))^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing

    if verbose
        println("*** Example file: worst-case performance of Kranoselskii-Mann iterations ***")
        println("\tPEPit guarantee:\t 1/4 ||xN - AxN||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_krasnoselskii_mann_increasing_step_sizes(3; verbose=true)
