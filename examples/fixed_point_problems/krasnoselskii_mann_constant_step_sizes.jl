using PEPit, OrderedCollections, Clarabel

function wc_krasnoselskii_mann_constant_step_sizes(n, gamma; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => 1.0)
    A = declare_function!(problem, LipschitzOperator, param)


    xs, _, _ = fixed_point!(A)
    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        x = (1 - gamma) * x + gamma * gradient!(A, x)
    end


    set_performance_metric!(problem, (1 / 2 * (x - gradient!(A, x)))^2)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if 1 / 2 <= gamma <= 1 / 2 * (1 + sqrt(n / (n + 1)))
        theoretical_tau = 1 / (n + 1) * (n / (n + 1))^n / (4 * gamma * (1 - gamma))
    elseif 1 / 2 * (1 + sqrt(n / (n + 1))) < gamma <= 1
        theoretical_tau = (2 * gamma - 1)^(2 * n)
    else
        error("$(gamma) is not a valid value for the step-size 'gamma'." *
              " 'gamma' must be a number between 1/2 and 1")
    end

    if verbose
        println("*** Example file: worst-case performance of Kranoselskii-Mann iterations ***")
        println("\tPEPit guarantee:\t 1/4||xN - AxN||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t 1/4||xN - AxN||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_krasnoselskii_mann_constant_step_sizes(3, 3 / 4; verbose=true)
