using PEPit
using OrderedCollections

function wc_improved_interior_algorithm(L, mu, c, lam, n; verbose=true)
    problem = PEP()


    func1 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)
    func2 = declare_function!(problem, ConvexIndicatorFunction, OrderedDict("D" => Inf); reuse_gradient=false)
    h = declare_function!(problem, StronglyConvexFunction, OrderedDict("mu" => mu); reuse_gradient=true)


    func = func1 + func2


    xs = stationary_point!(func)
    fs = value!(func, xs)
    ghs, hs = oracle!(h, xs)


    x0 = set_initial_point!(problem)
    gh0, h0 = oracle!(h, x0)
    g10, f10 = oracle!(func1, x0)


    x = x0
    z = x0
    g = g10
    gh = gh0
    ck = c
    for k in 0:(n - 1)
        alphak = (sqrt((ck * lam)^2 + 4 * ck * lam) - lam * ck) / 2
        ck = (1 - alphak) * ck
        y = (1 - alphak) * x + alphak * z
        if k >= 1
            g, _ = oracle!(func1, y)
        end
        z, _, _ = bregman_gradient_step!(g, gh, h + func2, alphak / ck)
        x = (1 - alphak) * x + alphak * z
        gh, _ = oracle!(h, z)
    end


    set_initial_condition!(problem, (hs - h0 - gh0 * (xs - x0)) * c + f10 - fs <= 1)


    set_performance_metric!(problem, value!(func, x) - fs)

    pepit_tau = solve!(problem; verbose=verbose)
    theoretical_tau = (4 * L) / (c * (n + 1)^2)

    if verbose
        println("*** Example file: worst-case performance of the Improved interior gradient algorithm in function values ***")
        println("\tPEPit guarantee:\t F(x_n)-F_* <= $(round(pepit_tau, digits=7)) (c * Dh(xs;x0) + f1(x0) - F_*)")
        println("\tTheoretical guarantee:\t F(x_n)-F_* <= $(round(theoretical_tau, digits=7)) (c * Dh(xs;x0) + f1(x0) - F_*)")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_improved_interior_algorithm(1.0, 1.0, 1.0, 1.0, 5; verbose=true)
