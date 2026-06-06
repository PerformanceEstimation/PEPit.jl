using PEPit, OrderedCollections

function wc_online_gradient_descent(M::Real, D::Real, n::Int; verbose = true)


    problem = PEP()


    gamma = D / (M * sqrt(n))


    fis = [declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M)) for _ in 1:n]


    h = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))


    F = sum(fis)


    x_ref = set_initial_point!(problem)
    x_ref, _, _ = proximal_step!(x_ref, h, 1)
    _, F_ref = oracle!(F, x_ref)


    x = set_initial_point!(problem)
    x, _, _ = proximal_step!(x, h, 1)


    f_saved = Vector{Expression}(undef, n)
    for i in 1:n
        g_i, f_i = oracle!(fis[i], x)
        f_saved[i] = f_i
        x, _, _ = proximal_step!(x - gamma * g_i, h, gamma)
    end


    set_performance_metric!(problem, sum(f_saved) - F_ref)


    pepit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = M * D * sqrt(n)


    if verbose != -1
        println("*** Example file: worst-case regret of online gradient descent for fixed step-sizes ***")
        println("\tPEPit guarantee:\t R_n <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t R_n <= $(round(theoretical_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


M, D, n = 1.0, 0.5, 2

pepit_tau, theoretical_tau = wc_online_gradient_descent(M, D, n; verbose=true)
