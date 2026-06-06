using PEPit
using OrderedCollections

function wc_inconsistent_halpern_iteration(n; verbose=true)
    problem = PEP()


    A = declare_function!(problem, NonexpansiveOperator, OrderedDict(); reuse_gradient=true)


    xs = Point()
    Txs = gradient!(A, xs)
    A.v = xs - Txs


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 0:(n - 1)
        x = 1 / (i + 2) * x0 + (1 - 1 / (i + 2)) * gradient!(A, x)
    end


    set_performance_metric!(problem, (x - gradient!(A, x) - A.v)^2)

    pepit_tau = solve!(problem; verbose=verbose)

    Hn = sum(1.0 / k for k in 1:n)
    theoretical_tau = ((sqrt(Hn + 12) + 1) / (n + 1))^2

    if verbose
        println("*** Example file: worst-case performance of (possibly inconsistent) Halpern Iterations ***")
        println("\tPEPit guarantee:\t ||xN - AxN - v||^2 <= $(round(pepit_tau, digits=6)) ||x0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||xN - AxN - v||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau

end


pepit_tau, theoretical_tau = wc_inconsistent_halpern_iteration(25; verbose=true)
