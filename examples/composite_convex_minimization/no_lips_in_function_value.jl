using PEPit, OrderedCollections, Clarabel

function wc_no_lips_in_function_value(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    d = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    func1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    h = (d + func1) / L

    func2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => Inf); reuse_gradient=false)

    func = func1 + func2


    xs = stationary_point!(func)
    ghs, hs = oracle!(h, xs)
    gfs, fs = oracle!(func1, xs)


    x0 = set_initial_point!(problem)
    gh0, h0 = oracle!(h, x0)
    gf0, f0 = oracle!(func1, x0)


    set_initial_condition!(problem, hs - h0 - gh0 * (xs - x0) <= 1)


    gfx = gf0
    ffx = f0
    ghx = gh0
    for i in 1:n
        x, _, _ = bregman_gradient_step!(gfx, ghx, func2 + h, gamma)
        gfx, ffx = oracle!(func1, x)
        gdx = gradient!(d, x)
        ghx = (gdx + gfx) / L
    end


    set_performance_metric!(problem, ffx - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 1 / (gamma * n)

    if verbose
        println("*** Example file: worst-case performance of the NoLips in function values ***")
        println("\tPEPit guarantee:\t F(x_n) - F_* <= $(round(pepit_tau, digits=6)) Dh(x_*; x_0)")
        println("\tTheoretical guarantee:\t F(x_n) - F_* <= $(round(theoretical_tau, digits=6)) Dh(x_*; x_0)")
    end

    return pepit_tau, theoretical_tau
end


L = 1.0
gamma = 1 / (2 * L)
pepit_tau, theoretical_tau = wc_no_lips_in_function_value(L, gamma, 3; verbose=true)
