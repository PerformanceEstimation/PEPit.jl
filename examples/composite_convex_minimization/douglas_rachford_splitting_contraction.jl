using PEPit
using OrderedCollections

function wc_douglas_rachford_splitting_contraction(mu, L, alpha, theta, n; verbose=true)
    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L); reuse_gradient=true)
    f2 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)


    w0 = set_initial_point!(problem)
    w0p = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w0p)^2 <= 1)


    w = w0
    for _ in 1:n
        x, _, _ = proximal_step!(w, f2, alpha)
        y, _, _ = proximal_step!(2 * x - w, f1, alpha)
        w = w + theta * (y - x)
    end


    wp = w0p
    for _ in 1:n
        xp, _, _ = proximal_step!(wp, f2, alpha)
        yp, _, _ = proximal_step!(2 * xp - wp, f1, alpha)
        wp = wp + theta * (yp - xp)
    end


    set_performance_metric!(problem, (w - wp)^2)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = nothing
    if theta == 1
        theoretical_tau = (max(1 / (1 + mu * alpha), (alpha * L) / (1 + alpha * L)))^(2n)
    end

    if verbose
        println("*** Example file: worst-case performance of the Douglas-Rachford splitting in distance ***")
        println("\tPEPit guarantee:\t ||w - wp||^2 <= $(round(PEPit_tau, digits=6)) ||w0 - w0p||^2")
        if theta == 1
            println("\tTheoretical guarantee:\t ||w - wp||^2 <= $(round(theoretical_tau, digits=6)) ||w0 - w0p||^2")
        end
    end

    return PEPit_tau, theoretical_tau
end


PEPit_tau, theoretical_tau = wc_douglas_rachford_splitting_contraction(0.1, 1.0, 3.0, 1.0, 2; verbose=true)
