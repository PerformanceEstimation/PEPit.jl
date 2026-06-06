using PEPit, OrderedCollections, Clarabel


function phi(mu, idx)
    if idx == -1
        return 0
    end
    return ((1 + 2 * mu)^(2 * idx + 2) - 1) / ((1 + 2 * mu)^2 - 1)
end

function wc_optimal_strongly_monotone_proximal_point(n, mu; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, StronglyMonotoneOperator, OrderedDict("mu" => mu))


    xs = stationary_point!(A)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x, y, y_prv = x0, x0, x0
    for i in 0:(n - 1)
        x_nxt, _, _ = proximal_step!(y, A, 1)
        y_nxt = x_nxt + (phi(mu, i) - 1) / phi(mu, i + 1) * (x_nxt - x) - 2 * mu * phi(mu, i) / phi(mu, i + 1) * (
                y - x_nxt) + (1 + 2 * mu) * phi(mu, i - 1) / phi(mu, i + 1) * (y_prv - x)
        x, y_prv, y = x_nxt, y, y_nxt
    end


    set_performance_metric!(problem, (y_prv - x)^2)


    pepit_verbose = verbose >= 0
    pepit_tau = solve!(problem; solver=solver, verbose=pepit_verbose)


    theoretical_tau = (2 * mu / ((1 + 2 * mu)^n - 1))^2


    if verbose != -1
        println("*** Example file: worst-case performance of Optimal Strongly-monotone Proximal Point Method ***")
        println("\tPEPit guarantee:\t ||AxN||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||AxN||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_optimal_strongly_monotone_proximal_point(10, 0.05; verbose=true)
