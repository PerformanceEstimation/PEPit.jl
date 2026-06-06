using PEPit, OrderedCollections, Clarabel

function wc_online_follow_regularized_leader(M::Real, D::Real, n::Int; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    M_list = [M for i in 1:n]
    eta = D / (M * sqrt(n))


    fis = [declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M_list[i])) for i in 1:n]


    h = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))


    F = sum(fis)


    x_ref = set_initial_point!(problem)
    x_ref, _, _ = proximal_step!(x_ref, h, 1)
    _, F_ref = oracle!(F, x_ref)


    x1 = set_initial_point!(problem)
    x1, _, _ = proximal_step!(x1, h, 1)


    f_occ = h
    x = x1
    f_saved = Vector{Expression}(undef, n)
    for i in 1:n
        g_i, f_i = oracle!(fis[i], x)
        f_saved[i] = f_i
        f_occ = f_occ + fis[i]
        if i < n
            x, _, _ = proximal_step!(x1, f_occ, eta)
        end
    end


    set_performance_metric!(problem, sum(f_saved) - F_ref)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = M * D * sqrt(n)


    if verbose != -1
        println("*** Example file: worst-case regret of online follow the regularized leader ***")
        println("\tPEPit guarantee:\t R_n <= $(round(pepit_tau, digits=6))")
        println("\tTheoretical guarantee:\t R_n <= $(round(theoretical_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


M, D, n = 1.0, 0.5, 2

pepit_tau, theoretical_tau = wc_online_follow_regularized_leader(M, D, n; solver=Clarabel.Optimizer, verbose=true)
