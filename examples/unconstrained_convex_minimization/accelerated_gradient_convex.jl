using PEPit, OrderedCollections, Mosek, MosekTools

function wc_accelerated_gradient_convex(mu, L, n; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    y = x0
    lam = 1.0
    lam_old = lam

    for _ in 1:n
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old^2 + 1)) / 2
        x_old = x
        x = y - 1 / L * gradient!(func, y)
        y = x + (lam_old - 1) / lam * (x - x_old)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * lam_old^2)

    mu != 0 && @warn "Momentum is tuned for non-strongly convex functions."


    if verbose
        println("*** Example file: worst-case performance of accelerated gradient method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_accelerated_gradient_convex(0, 1, 1; solver=Mosek.Optimizer, verbose=true)
