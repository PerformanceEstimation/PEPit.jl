using PEPit, OrderedCollections, Clarabel

function wc_no_lips_2(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    d1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    d2 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    func1 = (d2 - d1) / 2
    h = (d1 + d2) / L / 2
    func2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => Inf))


    func = func1 + func2


    x0 = set_initial_point!(problem)
    gh0, h0 = oracle!(h, x0)
    gf0, f0 = oracle!(func1, x0)
    _, F0 = oracle!(func, x0)


    x1, x2 = x0, x0
    gfx = gf0
    ghx = gh0
    hx1, hx2 = h0, h0
    for i in 1:n
        x2, _, _ = bregman_gradient_step!(gfx, ghx, func2 + h, gamma)
        gfx, _ = oracle!(func1, x2)
        ghx, hx2 = oracle!(h, x2)
        Dhx = hx1 - hx2 - ghx * (x1 - x2)

        x1, hx1 = x2, hx2

        set_performance_metric!(problem, Dhx)
    end
    _, Fx = oracle!(func, x2)


    set_initial_condition!(problem, F0 - Fx <= 1)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = gamma / n

    if verbose
        println("*** Example file: worst-case performance of the NoLips_2 in Bregman distance ***")
        println("\tPEPit guarantee:\t min_t Dh(x_(t-1), x_(t)) <= $(round(pepit_tau, digits=6)) (F(x_0) - F(x_n))")
        println("\tTheoretical guarantee:\t min_t Dh(x_(t-1), x_(t)) <= $(round(theoretical_tau, digits=6)) (F(x_0) - F(x_n))")
    end

    return pepit_tau, theoretical_tau
end


L = 1.0
gamma = 1 / L
pepit_tau, theoretical_tau = wc_no_lips_2(L, gamma, 3; verbose=true)
