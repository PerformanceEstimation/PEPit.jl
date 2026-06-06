using PEPit, OrderedCollections, Mosek, MosekTools

function wc_heavy_ball_momentum(mu, L, alpha, beta, n; solver=Mosek.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    f0 = value!(func, x0)


    set_initial_condition!(problem, (f0 - fs) <= 1)


    x_new = x0
    x_old = x0
    for _ in 1:n
        x_next = x_new - alpha * gradient!(func, x_new) + beta * (x_new - x_old)
        x_old = x_new
        x_new = x_next
    end


    set_performance_metric!(problem, value!(func, x_new) - fs)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)
    theoretical_tau = (1 - alpha * mu)^n

    if verbose
        println("*** Example file: worst-case performance of the Heavy-Ball method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0) - f(x_*))")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0) - f(x_*))")
    end

    return pepit_tau, theoretical_tau
end

mu = 0.1
L = 1.0
alpha = 1 / (2 * L)
beta = sqrt((1 - alpha * mu) * (1 - L * alpha))
pepit_tau, theoretical_tau = wc_heavy_ball_momentum(mu, L, alpha, beta, 2; verbose=true)
