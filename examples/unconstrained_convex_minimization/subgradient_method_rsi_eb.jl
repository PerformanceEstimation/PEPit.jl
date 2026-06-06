using PEPit, OrderedCollections, Clarabel

function wc_subgradient_method_rsi_eb(mu, L, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, RsiEbFunction, param)


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for _ in 1:n
        x = x - gamma * gradient!(func, x)
    end


    set_performance_metric!(problem, (x - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 - 2 * gamma * mu + gamma^2 * L^2)^n

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


mu = 0.1
L = 1.0
pepit_tau, theoretical_tau = wc_subgradient_method_rsi_eb(mu, L, mu / L^2, 4; solver=Clarabel.Optimizer, verbose=true)
