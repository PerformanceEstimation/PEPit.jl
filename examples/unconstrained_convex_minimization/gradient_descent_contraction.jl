using PEPit, OrderedCollections, Mosek, MosekTools

function wc_gradient_descent_contraction(L, mu, gamma, n; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))


    x0 = set_initial_point!(problem)
    y0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - y0)^2 <= 1)


    x = x0
    y = y0
    for _ in 1:n
        x = x - gamma * gradient!(func, x)
        y = y - gamma * gradient!(func, y)
    end


    set_performance_metric!(problem, (x - y)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max((1 - gamma * L)^2, (1 - gamma * mu)^2)^n

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-sizes in contraction ***")
        println("\tPEPit guarantee:\t ||x_n - y_n||^2 <= $(round(pepit_tau, digits=6)) ||x_0 - y_0||^2")
        println("\tTheoretical guarantee:\t ||x_n - y_n||^2 <= $(round(theoretical_tau, digits=6)) ||x_0 - y_0||^2")
    end

    return pepit_tau, theoretical_tau
end


L = 1.0
mu = 0.1
gamma = 1 / L
n = 1
pepit_tau, theoretical_tau = wc_gradient_descent_contraction(L, mu, gamma, n; verbose=true)
