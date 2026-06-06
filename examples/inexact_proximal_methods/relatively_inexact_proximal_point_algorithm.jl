using PEPit, OrderedCollections, Clarabel, OffsetArrays


function wc_relatively_inexact_proximal_point_algorithm(n, gamma, sigma; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(f)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = OffsetVector([x0 for _ in 0:n], 0:n)


    for i in 0:n-1

        x[i + 1], _, fx, _, _, _, epsVar = inexact_proximal_step!(x[i], f, gamma; opt="PD_gapII")


        add_constraint!(f, epsVar <= (sigma * (x[i + 1] - x[i]))^2 / 2)
    end


    set_performance_metric!(problem, value!(f, x[n]) - value!(f, xs))


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 + sigma) / (4 * gamma * n^sqrt(1 - sigma^2))


    if verbose

        println("*** Example file: worst-case performance of an inexact proximal point method in distance in function values ***")
        println("\tPEPit guarantee:\t f(x_n) - f(x_*) <= $(round(pepit_tau, digits=7)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n) - f(x_*) <= $(round(theoretical_tau, digits=7)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_relatively_inexact_proximal_point_algorithm(8, 10, 0.65; verbose=true)
