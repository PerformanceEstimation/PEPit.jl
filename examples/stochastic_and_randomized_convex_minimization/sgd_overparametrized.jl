using PEPit, OrderedCollections, Clarabel

function wc_sgd_overparametrized(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    fn = [declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu)) for _ in 1:n]
    func = sum(fn) / n


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    var = sum(gradient!(f, xs)^2 for f in fn) / n
    add_constraint!(problem, var <= 0.0)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    distavg = sum((x0 - gamma * gradient!(f, x0) - xs)^2 for f in fn) / n

    set_performance_metric!(problem, distavg)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)

    theoretical_tau = max(1 - gamma * mu, L * gamma - 1)^2

    if verbose
        println("*** Example file: worst-case performance of stochastic gradient descent with fixed step-size and with zero variance at the optimal point ***")
        println("\tPEPit guarantee:\t E[||x_1 - x_*||^2] <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t E[||x_1 - x_*||^2] <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_sgd_overparametrized(1.0, 0.1, 2.3, 5; solver=Clarabel.Optimizer, verbose=true)
