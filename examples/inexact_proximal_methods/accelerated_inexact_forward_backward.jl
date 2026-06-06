using PEPit, OrderedCollections


function wc_accelerated_inexact_forward_backward(L, zeta, n; verbose=true)


    problem = PEP()


    f = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)
    h = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)


    F = f + h


    xs = stationary_point!(F)


    Fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    gamma = 1 / L


    eta = (1 - zeta^2) * gamma


    A = 0.0


    x = x0
    z = x0


    hx = nothing


    for _ in 1:n

        A_next = A + (eta + sqrt(eta^2 + 4 * eta * A)) / 2


        y = x + (1 - A / A_next) * (z - x)


        gy = gradient!(f, y)


        x, _, hx, _, vx, _, eps_var = inexact_proximal_step!(y - gamma * gy, h, gamma; opt="PD_gapI")


        add_constraint!(h, eps_var <= (zeta * gamma)^2 / 2 * (vx + gy)^2)


        z = z - (A_next - A) * (vx + gy)


        A = A_next
    end


    set_performance_metric!(problem, value!(f, x) + hx - Fs)


    pepit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = 2 * L / (1 - zeta^2) / n^2


    if verbose

        println("*** Example file: worst-case performance of an inexact accelerated forward backward method ***")
        println("\tPEPit guarantee:\t F(x_n)-F_* <= $(round(pepit_tau, digits=7)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t F(x_n)-F_* <= $(round(theoretical_tau, digits=7)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_inexact_forward_backward(1.3, 0.45, 11; verbose=true)
