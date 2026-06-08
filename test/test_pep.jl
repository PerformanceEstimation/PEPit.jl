using Test
using JuMP: value

@testset "PEP Tests" begin

    PEPit.Point_counter[] = 0
    PEPit.Expression_counter[] = 0
    PEPit.Function_counter[] = 0
    PEPit.Global_Constraint_counter[] = 0
    PEPit.NEXT_ID[] = 0


    L = 1.0
    mu = 0.1
    gamma = 1 / L


    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x1 = x0 - gamma * gradient!(func, x0)


    set_performance_metric!(problem, (x1 - xs)^2)


    theoretical_tau = max((1 - mu * gamma)^2, (1 - L * gamma)^2)

    @testset "Instance Type Tests" begin
        @test problem isa PEP
        @test length(problem.list_of_functions) == 1
        @test length(problem.list_of_points) == 1
        @test length(problem.list_of_conditions) == 1
        @test length(problem.list_of_performance_metrics) == 1
        @test length(func._PEPit_func.list_of_constraints) == 0

        PEPit_tau = solve!(problem; verbose=false)
        @test length(func._PEPit_func.list_of_constraints) == 2
        @test PEPit.Point_counter[] == 3
        @test PEPit.Expression_counter[] == 2
        @test PEPit.Function_counter[] == 1
    end

    @testset "Eval Points and Function Values" begin
        solve!(problem; verbose=false)

        for triplet in func._PEPit_func.list_of_points
            point, gradient, function_value = triplet

            @test PEPit.evaluate(point) isa Vector{Float64}
            @test PEPit.evaluate(gradient) isa Vector{Float64}
            @test PEPit.evaluate(function_value) isa Float64
        end
    end

    @testset "Eval Points and Expression Values Defined Independently" begin

        x2, dx1, _ = inexact_gradient_step!(x1, func, gamma, 0.1; notion="absolute")
        set_performance_metric!(problem, (x2 - xs)^2)

        solve!(problem; verbose=false)

        @test PEPit.evaluate(x1) isa Vector{Float64}

        @test PEPit.evaluate(dx1) isa Vector{Float64}
        @test PEPit.evaluate(x2) isa Vector{Float64}
    end

end

@testset "PEP Duals and LMIs" begin

    PEPit.Point_counter[] = 0
    PEPit.Expression_counter[] = 0
    PEPit.Function_counter[] = 0
    PEPit.Global_Constraint_counter[] = 0
    PEPit.PSDMatrix_counter[] = 0
    PEPit.NEXT_ID[] = 0


    L = 1.0
    mu = 0.1
    gamma = 1 / L

    problem = PEP()
    func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))

    xs = stationary_point!(func)
    x0 = set_initial_point!(problem)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)

    x1 = x0 - gamma * gradient!(func, x0)
    set_performance_metric!(problem, (x1 - xs)^2)

    theoretical_tau = max((1 - mu * gamma)^2, (1 - L * gamma)^2)


    PEPit_tau = solve!(problem; verbose=false)
    @test isapprox(PEPit_tau, theoretical_tau; rtol=1e-3)

    for cond in problem.list_of_conditions
        @test cond._dual_variable_value isa Float64
        @test isapprox(cond._dual_variable_value, PEPit_tau; rtol=1e-3)
    end

    class_rhs = 2 * gamma * max(abs(1 - mu * gamma), abs(1 - L * gamma))
    for c in func._PEPit_func.list_of_constraints
        @test c._dual_variable_value isa Float64
        @test isapprox(c._dual_variable_value, class_rhs; rtol=1e-3)
    end


    R = 3.0
    problem.list_of_conditions = [(x0 - xs)^2 <= R^2]

    expr = Expression()
    matrix_of_expressions = [(x0-xs)^2 expr;
        expr 1]
    add_psd_matrix!(problem, matrix_of_expressions)

    problem.list_of_performance_metrics = [expr]
    tau_psd = solve!(problem; verbose=false)

    @test isapprox(tau_psd, R; rtol=1e-3)
    @test isapprox(PEPit.evaluate(expr), tau_psd; rtol=1e-3)


    problem2 = PEP()
    func2 = declare_function!(problem2, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    xs2 = stationary_point!(func2)
    x0_2 = set_initial_point!(problem2)
    set_initial_condition!(problem2, (x0_2 - xs2)^2 <= 1)
    x1_2 = x0_2 - gamma * gradient!(func2, x0_2)

    point = Point()
    expr2 = point^2
    matrix2 = [(x1_2-xs2)^2 expr2;
        expr2 1]
    add_psd_matrix!(problem2, matrix2)

    problem2.list_of_performance_metrics = [expr2]
    tau2 = solve!(problem2; verbose=false)

    theoretical_tau_sqrt = sqrt(max((1 - mu * gamma)^2, (1 - L * gamma)^2))
    @test isapprox(tau2, theoretical_tau_sqrt; rtol=1e-3)

    @test isapprox(sum(PEPit.evaluate(point) .^ 2), PEPit.evaluate(expr2); rtol=1e-3)
    @test isapprox(PEPit.evaluate(expr2), tau2; rtol=1e-3)


    Mval = PEPit.evaluate(problem2.list_of_psd[1])
    for i in 1:2, j in 1:2
        @test isapprox(Mval[i, j], tau2^(4 - i - j); rtol=1e-3)
    end


    Mdual = eval_dual(problem2.list_of_psd[1])
    for i in 1:2, j in 1:2
        expected = -0.5 * (-tau2)^(i + j - 3)
        @test isapprox(Mdual[i, j], expected; rtol=1e-3)
    end


    problem3 = PEP()
    func3 = declare_function!(problem3, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    xs3 = stationary_point!(func3)
    x0_3 = set_initial_point!(problem3)
    set_initial_condition!(problem3, (x0_3 - xs3)^2 <= 1)
    x1_3 = x0_3 - gamma * gradient!(func3, x0_3)
    set_performance_metric!(problem3, (x1_3 - xs3)^2)

    PEPit_tau3 = solve!(problem3; verbose=false)
    res_full = solve!(problem3; verbose=false, return_full_model=true)
    @test isapprox(res_full.wc_value, PEPit_tau3; atol=1e-2)


    res_trace = solve!(problem3; verbose=false, return_full_model=true, tracetrick=true)
    tau_trace = solve!(problem3; verbose=false, tracetrick=true)

    @test isapprox(tau_trace, PEPit_tau3; atol=1e-2)


end


@testset "PEP: dimension reduction with logdet" begin

    PEPit.Point_counter[] = 0
    PEPit.Expression_counter[] = 0
    PEPit.Function_counter[] = 0
    PEPit.Global_Constraint_counter[] = 0
    PEPit.NEXT_ID[] = 0

    L = 1.0
    mu = 0.1
    gamma = 1 / L

    build_problem() = begin
        p = PEP()
        f = declare_function!(p, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
        xs = stationary_point!(f)
        x0 = set_initial_point!(p)
        set_initial_condition!(p, (x0 - xs)^2 <= 1)
        x1 = x0 - gamma * gradient!(f, x0)
        set_performance_metric!(p, (x1 - xs)^2)
        p
    end


    pb1 = build_problem()
    τ = solve!(pb1; verbose=false)


    pb2 = build_problem()
    τ_logdet2 = solve!(pb2; verbose=false, logdetiters=2)
    @test isapprox(τ_logdet2, τ; atol=1e-6, rtol=1e-3)


    res_base = solve!(build_problem(); verbose=false, return_full_model=true)
    res_logd = solve!(build_problem(); verbose=false, return_full_model=true, logdetiters=2)
    nb_base, _, _ = PEPit._get_nb_eigs_and_corrected(value.(res_base.variables.G))
    nb_logd, _, _ = PEPit._get_nb_eigs_and_corrected(value.(res_logd.variables.G))
    @test nb_logd <= nb_base
end
