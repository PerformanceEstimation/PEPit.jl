using PEPit, OrderedCollections, Mosek, MosekTools

function wc_gradient_descent_quadratics(mu, L, gamma, n; solver=Mosek.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexQuadraticFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for _ in 1:n
        x = x - gamma * gradient!(func, x)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    t = 1 / (L * gamma * (2 * n + 1))
    if t < mu / L
        alpha = mu / L
    elseif t > 1
        alpha = 1
    else
        alpha = t
    end

    theoretical_tau = 0.5 * L * max(alpha * (1 - alpha * L * gamma)^(2 * n), (1 - L * gamma)^(2 * n))

    if verbose
        println("*** Example file: worst-case performance of gradient descent on quadratics with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


L = 3.0
mu = 0.3
pepit_tau, theoretical_tau = wc_gradient_descent_quadratics(mu, L, 1 / L, 4; verbose=true)
