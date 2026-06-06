using PEPit, OrderedCollections, Clarabel, OffsetArrays


function wc_accelerated_douglas_rachford_splitting(mu, L, alpha, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    func1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)
    func2 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L); reuse_gradient=true)


    func = func1 + func2


    xs = stationary_point!(func)
    fs = value!(func, xs)
    g1s, _ = oracle!(func1, xs)
    g2s, _ = oracle!(func2, xs)


    x0 = set_initial_point!(problem)


    theta = (1 - alpha * L) / (1 + alpha * L)


    ws = xs + alpha * g2s
    set_initial_condition!(problem, (ws - x0)^2 <= 1)


    x = OffsetVector([x0 for _ in 0:(n - 1)], 0:(n - 1))
    w = OffsetVector([x0 for _ in 0:n], 0:n)
    u = OffsetVector([x0 for _ in 0:n], 0:n)


    local y
    local fy


    for i in 0:(n - 1)

        x[i], _, _ = proximal_step!(u[i], func2, alpha)


        y, _, fy = proximal_step!(2 * x[i] - u[i], func1, alpha)


        w[i + 1] = u[i] + theta * (y - x[i])


        if i >= 1
            u[i + 1] = w[i + 1] + (i - 1) / (i + 2) * (w[i + 1] - w[i])
        else
            u[i + 1] = w[i + 1]
        end
    end


    set_performance_metric!(problem, value!(func2, y) + fy - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if alpha < 1 / L
        theoretical_tau = 2 / (alpha * theta * (n + 3)^2)
    else
        theoretical_tau = nothing
    end


    if verbose
        println("*** Example file: worst-case performance of the Accelerated Douglas Rachford Splitting in function values ***")
        println("\tPEPit guarantee:\t\t\t F(y_n)-F_* <= $(round(pepit_tau, digits=6)) ||x0 - ws||^2")
        if alpha < 1 / L
            println("\tTheoretical guarantee for quadratics:\t F(y_n)-F_* <= $(round(theoretical_tau, digits=6)) ||x0 - ws||^2")
        end
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_douglas_rachford_splitting(0.1, 1.0, 0.9, 2; verbose=true)
