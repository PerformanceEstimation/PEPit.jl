using PEPit, OrderedCollections


function wc_three_operator_splitting(mu1, L1, L3, alpha, theta, n; verbose=true)


    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu1, "L" => L1); reuse_gradient=true)
    f2 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)
    f3 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L3); reuse_gradient=true)


    w0 = set_initial_point!(problem)
    w0p = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w0p)^2 <= 1)


    w = w0
    for _ in 1:n

        x, _, _ = proximal_step!(w, f2, alpha)


        gx, _ = oracle!(f3, x)


        y, _, _ = proximal_step!(2 * x - w - alpha * gx, f1, alpha)


        w = w + theta * (y - x)
    end


    wp = w0p
    for _ in 1:n

        xp, _, _ = proximal_step!(wp, f2, alpha)


        gxp, _ = oracle!(f3, xp)


        yp, _, _ = proximal_step!(2 * xp - wp - alpha * gxp, f1, alpha)


        wp = wp + theta * (yp - xp)
    end


    set_performance_metric!(problem, (w - wp)^2)


    pepit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = nothing


    if verbose

        println("*** Example file: worst-case performance of the Three Operator Splitting in distance ***")
        println("\tPEPit guarantee:\t ||w^1_n - w^0_n||^2 <= $(round(pepit_tau, digits=6)) ||w^1_0 - w^0_0||^2")
    end


    return pepit_tau, theoretical_tau
end


L3 = 1.0


alpha = 1 / L3


pepit_tau, theoretical_tau = wc_three_operator_splitting(0.1, 10.0, L3, alpha, 1.0, 4; verbose=true)
