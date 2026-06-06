using PEPit, OrderedCollections, Clarabel, OffsetArrays

function wc_accelerated_proximal_point(alpha::Real, n::Int; solver=Clarabel.Optimizer, verbose::Int=1)

    problem = PEP()


    A = declare_function!(problem, MonotoneOperator, OrderedDict())


    xs = stationary_point!(A)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = OffsetVector(fill(x0, n + 1), 0:n)
    y = OffsetVector(fill(x0, n + 1), 0:n)

    for i in 0:(n - 2)
        x[i + 1], _, _ = proximal_step!(y[i + 1], A, alpha)
        y[i + 2] = x[i + 1] + i / (i + 2) * (x[i + 1] - x[i]) - i / (i + 2) * (x[i] - y[i])
    end
    x[n], _, _ = proximal_step!(y[n], A, alpha)


    set_performance_metric!(problem, (x[n] - y[n])^2)


    pepit_verbose = verbose >= 0
    τ_PEPit = solve!(problem, solver=solver, verbose=pepit_verbose)


    τ_theory = 1 / (n^2)

    if verbose != -1
        @info "*** Example file: worst-case performance of the Accelerated Proximal Point Method***"
        @info "PEPit guarantee:\t ||x_n - y_n||^2 <= $(round(τ_PEPit, digits=6)) ||x_0 - x_s||^2"
        @info "Theoretical guarantee:\t ||x_n - y_n||^2 <= $(round(τ_theory, digits=6)) ||x_0 - x_s||^2"
    end

    return τ_PEPit, τ_theory
end

τ_PEPit, τ_theory =
wc_accelerated_proximal_point(2.0, 10; verbose=1)
