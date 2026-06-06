using PEPit
using OrderedCollections

function wc_sgd(L, mu, gamma, v, R, n; verbose=true)
    problem = PEP()


    fn = [declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu); reuse_gradient=true) for _ in 1:n]
    func = sum(fn) / n


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    var = sum(gradient!(f, xs)^2 for f in fn) / n
    add_constraint!(problem, var <= v^2)
    set_initial_condition!(problem, (x0 - xs)^2 <= R^2)


    distavg = sum((x0 - gamma * gradient!(f, x0) - xs)^2 for f in fn) / n

    set_performance_metric!(problem, distavg)

    pepit_tau = solve!(problem; verbose=verbose)

    theoretical_tau = (gamma * R * (L - mu) / 2 + sqrt(R^2 * (L + mu)^2 / 4 * (gamma - 2 / (L + mu))^2 + v^2 * gamma^2))^2

    if verbose
        println("*** Example file: worst-case performance of stochastic gradient descent with fixed step-size ***")
        println("\tPEPit guarantee:\t E[||x_1 - x_*||^2] <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t E[||x_1 - x_*||^2] <= $(round(theoretical_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_sgd(1.0, 0.1, 0.7, 1.0, 2.0, 5; verbose=true)
