using PEPit
using OrderedCollections

function wc_gradient_descent(L, gamma, n; verbose=true)
    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothFunction, param; reuse_gradient=true)


    x0 = set_initial_point!(problem)
    g0, f0 = oracle!(func, x0)


    x = x0
    gx, fx = g0, f0
    set_performance_metric!(problem, gx^2)


    for i in 1:n
        x = x - gamma * gx
        gx, fx = oracle!(func, x)
        set_performance_metric!(problem, gx^2)
    end


    set_initial_condition!(problem, f0 - fx <= 1)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = 4 / 3 * L / n

    if verbose
        println("*** Example file: worst-case performance of gradient descent with fixed step-size ***")
        println("\tPEPit guarantee:\t min_i ||f'(x_i)||^2 <= $(round(PEPit_tau, digits=6)) (f(x_0)-f_*)")
        println("\tTheoretical guarantee:\t min_i ||f'(x_i)||^2 <= $(round(theoretical_tau, digits=6)) (f(x_0)-f_*)")
    end

    return PEPit_tau, theoretical_tau
end


PEPit_val, theoretical_val = wc_gradient_descent(1.0, 1.0, 5; verbose=true)
