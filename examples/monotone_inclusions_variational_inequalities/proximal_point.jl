using PEPit, OrderedCollections, Clarabel

function wc_proximal_point(alpha::Real, n::Int; solver=Clarabel.Optimizer, verbose::Int=1)

    problem = PEP()


    A = declare_function!(problem, MonotoneOperator, OrderedDict())


    xs = stationary_point!(A)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    previous_x = x0
    for _ in 1:n
        previous_x = x
        x, _, _ = proximal_step!(previous_x, A, alpha)
    end


    set_performance_metric!(problem, (x - previous_x)^2)


    pepit_verbose = verbose >= 0
    τ_PEPit = solve!(problem, solver=solver, verbose=pepit_verbose)


    τ_theory = (1 - 1 / n)^(n - 1) / n

    if verbose != -1
        @info "*** Example file: worst-case performance of the Proximal Point Method***"
        @info "PEPit guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(τ_PEPit, digits=6)) ||x0 - xs||^2"
        @info "Theoretical guarantee:\t ||x(n) - x(n-1)||^2 <= $(round(τ_theory, digits=6)) ||x0 - xs||^2"
    end

    return τ_PEPit, τ_theory
end

τ_PEPit, τ_theory =
wc_proximal_point(2.0, 10; verbose=1)
