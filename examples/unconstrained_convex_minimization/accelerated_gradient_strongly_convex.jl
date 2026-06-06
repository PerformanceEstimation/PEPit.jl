using PEPit
using OrderedCollections
using JuMP

function wc_accelerated_gradient_strongly_convex(mu, L, n; verbose=false)
    problem = PEP()
    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param; reuse_gradient=true)

    xs = stationary_point!(func)
    fs = value!(func, xs)
    x0 = set_initial_point!(problem)

    set_initial_condition!(problem, value!(func, x0) - fs + mu / 2 * (x0 - xs)^2 <= 1)

    kappa = mu / L
    x_new, y = x0, x0
    for i in 1:n
        x_old = x_new
        x_new = y - 1 / L * gradient!(func, y)
        y = x_new + (1 - sqrt(kappa)) / (1 + sqrt(kappa)) * (x_new - x_old)
    end

    set_performance_metric!(problem, value!(func, x_new) - fs)

    PEPit_tau = solve!(problem, verbose=verbose)
    theoretical_tau = (1 - sqrt(kappa))^n
    mu == 0 && @warn "Momentum is tuned for strongly convex functions!"

    if verbose
        @info "🐱 Example file: worst-case performance of the accelerated gradient method"
        @info "💻  PEPit guarantee: f(x_n)-f_*  <= $(round(PEPit_tau, digits=6)) (f(x_0) - f(x_*) + mu/2*||x_0 - x_*||^2)"
        @info "📝 Theoretical guarantee: f(x_n)-f_*  <= $(round(theoretical_tau, digits=6)) (f(x_0) - f(x_*) + mu/2*||x_0 - x_*||^2)"
    end

    @info "🐯 Detailed results"

    res = solve!(problem; verbose=false, return_full_model=true)

    @show res.wc_value
    @show res.model
    @show value.(res.variables.F)
    @show value.(res.variables.G)
    @show dual.(res.constraints.initial)
    @show dual.(res.constraints.class)
    @show dual.(res.constraints.performance)

    return PEPit_tau, theoretical_tau

end

PEPit_val, theoretical_val = wc_accelerated_gradient_strongly_convex(0.1, 1.0, 2, verbose=true)
