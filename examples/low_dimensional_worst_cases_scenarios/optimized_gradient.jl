using PEPit
using OrderedCollections

function wc_optimized_gradient(L, n; verbose=true)
    problem = PEP()


    func = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    theta_new = 1.0
    x_new = x0
    y = x0
    for i in 1:n
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        theta_old = theta_new
        if i < n
            theta_new = (1 + sqrt(4 * theta_new^2 + 1)) / 2
        else
            theta_new = (1 + sqrt(8 * theta_new^2 + 1)) / 2
        end
        y = x_new + (theta_old - 1) / theta_new * (x_new - x_old) + theta_old / theta_new * (x_new - y)
    end


    set_performance_metric!(problem, value!(func, y) - fs)

    pepit_tau_orig = solve!(problem; verbose=verbose, tracetrick=false)

    pepit_tau_trace_tricked = solve!(problem; verbose=verbose, tracetrick=true)

    theoretical_tau = L / (2 * theta_new^2)

    if verbose
        println("*** Example file: worst-case performance of optimized gradient method to see application of trace trick***")
        println("\tPEPit guarantee original:\t f(y_n)-f_* == $(round(pepit_tau_orig, digits=8)) ||x_0 - x_*||^2")
        println("\tPEPit guarantee after applying trace trick:\t f(y_n)-f_* == $(round(pepit_tau_trace_tricked, digits=8)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(y_n)-f_* <= $(round(theoretical_tau, digits=8)) ||x_0 - x_*||^2")
    end

    return pepit_tau_orig, theoretical_tau
end


pepit_tau, theoretical_tau = wc_optimized_gradient(3.0, 4; verbose=true)
