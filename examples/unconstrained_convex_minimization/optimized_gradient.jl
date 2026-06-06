using PEPit, OrderedCollections, Mosek, MosekTools

function wc_optimized_gradient(L, n; solver=Mosek.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    theta_new = 1.0
    x_new = x0
    y = x0
    for i in 0:(n - 1)
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        theta_old = theta_new
        if i < n - 1
            theta_new = (1 + sqrt(4 * theta_new^2 + 1)) / 2
        else
            theta_new = (1 + sqrt(8 * theta_new^2 + 1)) / 2
        end

        y = x_new + (theta_old - 1) / theta_new * (x_new - x_old) + theta_old / theta_new * (x_new - y)
    end


    set_performance_metric!(problem, value!(func, y) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * theta_new^2)


    if verbose
        println("*** Example file: worst-case performance of optimized gradient method ***")
        println("\tPEPit guarantee:\t f(y_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(y_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_optimized_gradient(3.0, 4; verbose=true)
