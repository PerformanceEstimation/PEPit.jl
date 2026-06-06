using PEPit, OrderedCollections, Clarabel, OffsetArrays

function wc_optimized_gradient_for_gradient(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)
    f0 = value!(func, x0)


    set_initial_condition!(problem, f0 - fs <= 1)


    theta_tmp = [1.0]
    for i in 0:(n - 1)
        if i < n - 1
            push!(theta_tmp, (1 + sqrt(4 * theta_tmp[i + 1]^2 + 1)) / 2)
        else
            push!(theta_tmp, (1 + sqrt(8 * theta_tmp[i + 1]^2 + 1)) / 2)
        end
    end
    reverse!(theta_tmp)
    theta_tilde = OffsetVector(theta_tmp, 0:n)


    x = x0
    y_new = x0

    for i in 0:(n - 1)
        y_old = y_new
        y_new = x - 1 / L * gradient!(func, x)
        x = y_new + (theta_tilde[i] - 1) * (2 * theta_tilde[i + 1] - 1) / theta_tilde[i] / (2 * theta_tilde[i] - 1) *
            (y_new - y_old) + (2 * theta_tilde[i + 1] - 1) / (2 * theta_tilde[i] - 1) * (y_new - x)
    end


    set_performance_metric!(problem, gradient!(func, x)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 * L / (theta_tilde[0]^2)


    if verbose
        println("*** Example file: worst-case performance of optimized gradient method for gradient ***")
        println("\tPEP-it guarantee:\t ||f'(x_n)||^2 <= $(round(pepit_tau, digits=6)) (f(x_0) - f_*)")
        println("\tTheoretical guarantee:\t ||f'(x_n)||^2 <= $(round(theoretical_tau, digits=6)) (f(x_0) - f_*)")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_optimized_gradient_for_gradient(3.0, 4; solver=Clarabel.Optimizer, verbose=true)
