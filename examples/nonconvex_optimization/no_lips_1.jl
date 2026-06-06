using PEPit, OrderedCollections, Clarabel, OffsetArrays

function wc_no_lips_1(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    d1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    d2 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=true)
    func1 = (d2 - d1) / 2
    h = (d1 + d2) / 2 / L


    func2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => Inf); reuse_gradient=false)


    func = func1 + func2


    x0 = set_initial_point!(problem)
    gh0, h0 = oracle!(h, x0)
    gf0, f0 = oracle!(func1, x0)
    _, F0 = oracle!(func, x0)


    xx = OffsetVector([x0 for _ in 0:n], 0:n)
    gfx = gf0
    ghx = OffsetVector([gh0 for _ in 0:n], 0:n)
    hx = OffsetVector([h0 for _ in 0:n], 0:n)
    local Fx::Expression
    for i in 0:(n-1)
        xx[i + 1], _, _ = bregman_gradient_step!(gfx, ghx[i], func2 + h, gamma)
        gfx, _ = oracle!(func1, xx[i + 1])
        ghx[i + 1], hx[i + 1] = oracle!(h, xx[i + 1])
        Dh = hx[i + 1] - hx[i] - ghx[i] * (xx[i + 1] - xx[i])

        set_performance_metric!(problem, Dh)
    end
    _, Fx = oracle!(func, xx[n])


    set_initial_condition!(problem, F0 - Fx <= 1)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = gamma / (n * (1 - L * gamma))


    if verbose
        println("*** Example file: worst-case performance of the NoLips in Bregman divergence ***")
        println("\tPEPit guarantee:\t min_t Dh(x_(t+1), x_(t)) <= $(round(pepit_tau, digits=6)) (F(x_0) - F(x_n))")
        println("\tTheoretical guarantee :\t min_t Dh(x_(t+1), x_(t)) <= $(round(theoretical_tau, digits=6)) (F(x_0) - F(x_n))")
    end


    return pepit_tau, theoretical_tau
end


L = 1.0
gamma = 1 / (2 * L)
pepit_tau, theoretical_tau = wc_no_lips_1(L, gamma, 5; verbose=true)
