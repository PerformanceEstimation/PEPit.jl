using PEPit, OrderedCollections, Clarabel


function wc_accelerated_proximal_gradient(mu, L, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    h = declare_function!(problem, ConvexFunction, OrderedDict())
    F = f + h


    xs = stationary_point!(F)
    Fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    y = x0
    lam = 1
    lam_old = lam
    local hx_new::Expression
    for i in 0:(n-1)

        lam_old = lam
        lam = (1 + sqrt(4 * lam_old^2 + 1)) / 2


        x_old = x_new


        x_new, _, hx_new = proximal_step!(y - 1 / L * gradient!(f, y), h, 1 / L)


        y = x_new + (lam_old - 1) / lam * (x_new - x_old)
    end


    set_performance_metric!(problem, (value!(f, x_new) + hx_new) - Fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * lam_old^2)

    if mu != 0
        println("Warning: momentum is tuned for non-strongly convex functions.")
    end


    if verbose
        println("*** Example file: worst-case performance of the Accelerated Proximal Gradient Method in function values***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_proximal_gradient(0.0, 1.0, 4; solver=Clarabel.Optimizer, verbose=true)
