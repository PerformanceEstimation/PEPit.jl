using PEPit
using OrderedCollections
using OffsetArrays

function wc_douglas_rachford_splitting(L, alpha, theta, n; verbose=true)

    problem = PEP()


    f1 = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)
    f2 = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)
    F  = f1 + f2


    xs = stationary_point!(F)
    fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    iters = 0:n-1
    x = OffsetVector(fill(x0, n), iters)
    w = OffsetVector(fill(x0, n + 1), 0:n)


    local y::Point
    local fy::Expression


    for i in iters
        xi, _, _ = proximal_step!(w[i], f2, alpha)
        x[i] = xi
        y, _, fy = proximal_step!(2 * x[i] - w[i], f1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])
    end


    set_initial_condition!(problem, (x[0] - xs)^2 <= 1)


    set_performance_metric!(problem, value!(f2, y) + fy - fs)


    PEPit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = nothing
    if theta == 1 && alpha == 1 && L == 1 && 1 <= n <= 10
        pesto_tau = OffsetVector([1/4, 0.1273, 0.0838, 0.0627, 0.0501,
                                  0.0417, 0.0357, 0.0313, 0.0278, 0.0250], 0:9)
        theoretical_tau = pesto_tau[n - 1]
    end

    if verbose
        println("*** Example file: worst-case performance of the Douglas–Rachford Splitting in function values ***")
        println("\tPEPit guarantee:\t f(y_n)-f_* <= $(round(PEPit_tau, digits=4)) ||x[0] - xs||^2")
        if theoretical_tau !== nothing
            println("\tTheoretical guarantee:\t f(y_n)-f_* <= $(round(theoretical_tau, digits=4)) ||x[0] - xs||^2")
        end
    end

    return PEPit_tau, theoretical_tau
end


PEPit_tau, theoretical_tau = wc_douglas_rachford_splitting(1.0, 1.0, 1.0, 9; verbose=true)
