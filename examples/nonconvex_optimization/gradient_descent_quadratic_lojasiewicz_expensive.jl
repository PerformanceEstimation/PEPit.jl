using PEPit, OrderedCollections, Clarabel

function wc_gradient_descent_quadratic_lojasiewicz_expensive(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L, "mu" => mu)
    func = declare_function!(problem, SmoothQuadraticLojasiewiczFunctionExpensive, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, value!(func, x0) - fs <= 1)


    x = x0
    for i in 1:n
        g = gradient!(func, x)
        x = x - gamma * g
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    m, mp = -L, mu
    if 0 <= gamma <= 1 / L
        theoretical_tau = (mp * (1 - L * gamma) + sqrt(
            (L - m) * (m - mp) * (2 - L * gamma) * mp * gamma + (L - m)^2)^2 / (L - m + mp)^2)
    elseif 1 / L <= gamma <= 3 / (m + L + sqrt(m^2 - L * m + L^2))
        theoretical_tau = ((L * gamma - 2) * (m * gamma - 2) * mp * gamma) / ((L + m - mp) * gamma - 2) + 1
    elseif 3 / (m + L + sqrt(m^2 - m * L + L^2)) <= gamma <= 2 / L
        theoretical_tau = (L * gamma - 1)^2 / ((L * gamma - 1)^2 + mp * gamma * (2 - L * gamma))
    else
        theoretical_tau = nothing
    end

    theoretical_tau = theoretical_tau^n

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-size ***")
        println("*** \t (smooth problem satisfying a Lojasiewicz inequality; expensive version) ***")
        println("\tPEPit guarantee:\t f(x_1) - f(x_*) <= $(round(pepit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t f(x_1) - f(x_*) <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return pepit_tau, theoretical_tau
end


L, mu, gamma, n = 1.0, 0.2, 1.0, 1
pepit_tau, theoretical_tau = wc_gradient_descent_quadratic_lojasiewicz_expensive(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)
