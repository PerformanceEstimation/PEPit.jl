using PEPit
using OrderedCollections

function wc_accelerated_gradient_method(L, gamma, lam; verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param; reuse_gradient=true)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xn = set_initial_point!(problem)
    _gn, fn = oracle!(func, xn)
    zn = set_initial_point!(problem)


    lam_np1 = (1 + sqrt(4 * lam^2 + 1)) / 2
    tau = 1 / lam_np1
    yn = (1 - tau) * xn + tau * zn
    gyn = gradient!(func, yn)

    eta = (lam_np1^2 - lam^2) / L
    znp1 = zn - eta * gyn

    xnp1 = yn - gamma * gyn
    _gnp1, fnp1 = oracle!(func, xnp1)


    final_lyapunov = lam_np1^2 * (fnp1 - fs) + L / 2 * (znp1 - xs)^2
    init_lyapunov = lam^2 * (fn - fs) + L / 2 * (zn - xs)^2

    set_performance_metric!(problem, final_lyapunov - init_lyapunov)

    pepit_tau = solve!(problem; verbose=verbose)

    theoretical_tau = gamma == 1 / L ? 0.0 : nothing

    if verbose
        println("*** Example file: worst-case performance of accelerated gradient method for a given Lyapunov function ***")
        println("\tPEPit guarantee:\t V_(n+1) - V_n <= $(round(pepit_tau, digits=6))")
        if gamma == 1 / L
            println("\tTheoretical guarantee:\t V_(n+1) - V_n <= $(round(theoretical_tau, digits=6))")
        end
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_gradient_method(1.0, 1.0, 10.0; verbose=true)
