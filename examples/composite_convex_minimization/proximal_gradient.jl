using PEPit, OrderedCollections, Clarabel


function wc_proximal_gradient(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    f2 = declare_function!(problem, ConvexFunction, OrderedDict())
    func = f1 + f2


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for _ in 1:n

        y = x - gamma * gradient!(f1, x)

        x, _, _ = proximal_step!(y, f2, gamma)
    end


    set_performance_metric!(problem, (x - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max((1 - mu * gamma)^2, (1 - L * gamma)^2)^n


    if verbose
        println("*** Example file: worst-case performance of the Proximal Gradient Method in function values***")
        println("\tPEPit guarantee:\t ||x_n - x_*||^2 <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t ||x_n - x_*||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_proximal_gradient(1.0, 0.1, 1.0, 2; solver=Clarabel.Optimizer, verbose=true)
