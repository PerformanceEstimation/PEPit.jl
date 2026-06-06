using PEPit
using OrderedCollections

function wc_bregman_proximal_point(gamma, n; verbose=true)
    problem = PEP()


    f1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)
    f2 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)


    xs = stationary_point!(f1)
    fs = value!(f1, xs)
    gf2s, f2s = oracle!(f2, xs)


    x0 = set_initial_point!(problem)
    gf20, f20 = oracle!(f2, x0)


    set_initial_condition!(problem, f2s - f20 - gf20 * (xs - x0) <= 1)


    sx = gf20
    local f1_val::Expression
    for i in 1:n
        x, sx, f2x, g1x, f1_val = bregman_proximal_step!(sx, f2, f1, gamma)
    end


    set_performance_metric!(problem, f1_val - fs)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = 1 / (gamma * n)

    if verbose
        println("*** Example file: worst-case performance of the Bregman Proximal Point in function values ***")
        println("\tPEPit guarantee:\t F(x_n)-F_* <= $(round(PEPit_tau, digits=7)) Dh(x_*; x_0)")
        println("\tTheoretical guarantee:\t F(x_n)-F_* <= $(round(theoretical_tau, digits=7)) Dh(x_*; x_0)")
    end

    return PEPit_tau, theoretical_tau
end


PEPit_tau, theoretical_tau = wc_bregman_proximal_point(3.0, 5; verbose=true)
