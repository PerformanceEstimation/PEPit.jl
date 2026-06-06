using PEPit, OrderedCollections, Clarabel

function wc_no_lips_in_bregman_divergence(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    d = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    func1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    h = (d + func1) / L


    func2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => Inf); reuse_gradient=false)


    func = func1 + func2


    xs = stationary_point!(func)
    ghs, hs = oracle!(h, xs)


    x0 = set_initial_point!(problem)
    gh0, h0 = oracle!(h, x0)
    gf0, f0 = oracle!(func1, x0)


    set_initial_condition!(problem, hs - h0 - gh0 * (xs - x0) <= 1)


    x1, x2 = x0, x0
    gfx = gf0
    ghx = gh0
    hx1, hx2 = h0, h0
    for i in 0:(n-1)
        x2, _, _ = bregman_gradient_step!(gfx, ghx, func2 + h, gamma)
        gfx, _ = oracle!(func1, x2)
        ghx, hx2 = oracle!(h, x2)
        Dhx = hx1 - hx2 - ghx * (x1 - x2)

        x1 = x2
        hx1 = hx2

        set_performance_metric!(problem, Dhx)
    end


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 2 / (n * (n - 1))


    if verbose
        println("*** Example file: worst-case performance of the NoLips_2 in Bregman divergence ***")
        println("\tPEPit guarantee:\t min_t Dh(x_(t-1); x_t) <= $(round(pepit_tau, digits=6)) Dh(x_*; x_0)")
        println("\tTheoretical guarantee:\t min_t Dh(x_(t-1); x_t) <= $(round(theoretical_tau, digits=6)) Dh(x_*; x_0)")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
gamma = 1 / L
pepit_tau, theoretical_tau = wc_no_lips_in_bregman_divergence(L, gamma, 10; verbose=true)
