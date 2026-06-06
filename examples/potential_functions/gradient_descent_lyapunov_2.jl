using PEPit, OrderedCollections, Clarabel

function wc_gradient_descent_lyapunov_2(L, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    xn = set_initial_point!(problem)
    gn, fn = oracle!(func, xn)


    xnp1 = xn - gamma * gn
    gnp1, fnp1 = oracle!(func, xnp1)


    init_lyapunov = (2 * n + 1) * L * (fn - fs) + n * (n + 2) * gn^2 + L^2 * (xn - xs)^2
    final_lyapunov = (2 * n + 3) * L * (fnp1 - fs) + (n + 1) * (n + 3) * gnp1^2 + L^2 * (xnp1 - xs)^2


    set_performance_metric!(problem, final_lyapunov - init_lyapunov)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)

    theoretical_tau = gamma == 1 / L ? 0.0 : nothing

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step size for a given Lyapunov function ***")
        println("\tPEPit guarantee:\tV_(n+1) - V_(n) <= $(round(pepit_tau, digits=6))")
        if gamma == 1 / L
            println("\tTheoretical guarantee:\tV_(n+1) - V_(n) <= $(round(theoretical_tau, digits=6))")
        end
    end

    return pepit_tau, theoretical_tau
end


L = 1.0
pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_2(L, 1 / L, 10; verbose=true)
