using PEPit, OrderedCollections, Clarabel

function wc_gradient_descent_silver_stepsize_convex(L, n; solver=Clarabel.Optimizer, verbose=true)

    k = log2(n + 1)
    if !isinteger(k)
        @warn "Silver step-size strategy is only defined when n is a power of 2 minus 1." *
              " The provided input n is not a power of 2 minus 1." *
              " n is reset as the largest acceptable value smaller than the provided one."
        k = floor(Int, k)
        n = 2^k - 1
    end


    fast_dyadic_valuation(i) = trailing_zeros(i)
    h = [1 + (1 + sqrt(2))^(fast_dyadic_valuation(i) - 1) for i in 1:n]


    problem = PEP()


    param = OrderedDict("L" => L)
    func = declare_function!(problem, SmoothConvexFunction, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 1:n
        x = x - h[i] / L * gradient!(func, x)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (1 + sqrt(4 * (1 + sqrt(2))^(2 * k) - 3))


    if verbose
        println("*** Example file: worst-case performance of gradient descent with silver step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_convex(10.0, 7; solver=Clarabel.Optimizer, verbose=true)
