using PEPit, OrderedCollections, Clarabel

function wc_online_follow_leader(M::Real, D::Real, n::Int; solver = Clarabel.Optimizer, verbose = true)


    problem = PEP()


    M_list = [M for i in 1:n]


    fis = [declare_function!(problem, ConvexLipschitzFunction, OrderedDict("M" => M_list[i])) for i in 1:n]


    h = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => D))


    x_ref = set_initial_point!(problem)
    x_ref, _, _ = proximal_step!(x_ref, h, 1)


    x1 = set_initial_point!(problem)
    x1, _, _ = proximal_step!(x1, h, 1)


    x = x1
    x_saved = Vector{Point}(undef, n)
    g_saved = Vector{Point}(undef, n)
    f_saved = Vector{Expression}(undef, n)
    f_occ = h
    for i in 1:n
        x_saved[i] = x - x_ref
        g, f = oracle!(fis[i], x)
        f_saved[i] = f - value!(fis[i], x_ref)
        g_saved[i] = g

        f_occ = f_occ + fis[i]
        if i < n

            x = stationary_point!(f_occ)
        end
    end


    set_performance_metric!(problem, sum(f_saved))


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing


    if verbose != -1
        println("*** Example file: worst-case regret of online follow the leader ***")
        println("\tPEPit guarantee:\t R_n <= $(round(pepit_tau, digits=6))")
    end

    return pepit_tau, theoretical_tau
end


M, D, n = 1.0, 0.5, 2

pepit_tau, theoretical_tau = wc_online_follow_leader(M, D, n; verbose=true)
