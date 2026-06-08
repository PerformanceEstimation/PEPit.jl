using Test

@testset "Constraints" begin

    PEPit.Point_counter[] = 0
    PEPit.Expression_counter[] = 0
    PEPit.Function_counter[] = 0
    PEPit.Global_Constraint_counter[] = 0
    PEPit.NEXT_ID[] = 0


    L = 1.0
    mu = 0.1
    gamma = 1 / L


    problem = PEP()


    param = OrderedDict("L" => L, "mu" => mu)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x1 = x0 - gamma * gradient!(func, x0)


    set_performance_metric!(problem, (x1 - xs)^2)


    solution = solve!(problem; verbose=false)

    @testset "Instance Type Tests" begin
        @test func isa SmoothStronglyConvexFunction
        @test func._PEPit_func isa PEPFunction
        @test xs isa Point
        @test x0 isa Point
        @test x1 isa Point

        for constraint in problem.list_of_conditions
            @test constraint isa Constraint
            @test constraint.expression isa Expression
        end

        for constraint in func._PEPit_func.list_of_constraints
            @test constraint isa Constraint
            @test constraint.expression isa Expression
        end
    end

    @testset "Counter Tests" begin
        @test func._PEPit_func.counter == 0
        @test xs.counter == 0
        @test x0.counter == 1
        @test x1.counter === nothing


        for (i, constraint) in enumerate(problem.list_of_conditions)
            @test constraint.counter == i - 1
        end


        for (i, constraint) in enumerate(func._PEPit_func.list_of_constraints)
            @test constraint.counter == (i - 1) + length(problem.list_of_conditions)
        end
    end

    @testset "Equality/Inequality Tests" begin
        for constraint in func._PEPit_func.list_of_constraints
            @test constraint.equality_or_inequality isa String
            @test constraint.equality_or_inequality in ["equality", "inequality"]
        end

        for constraint in problem.list_of_conditions
            @test constraint.equality_or_inequality isa String
            @test constraint.equality_or_inequality in ["equality", "inequality"]
        end
    end

    @testset "Evaluation Tests" begin
        for constraint in func._PEPit_func.list_of_constraints
            @test PEPit.evaluate(constraint) isa Float64
        end

        for constraint in problem.list_of_conditions
            @test PEPit.evaluate(constraint) isa Float64
        end
    end

    @testset "Dual Evaluation Tests" begin
        for constraint in func._PEPit_func.list_of_constraints
            @test eval_dual(constraint) isa Float64
        end

        for constraint in problem.list_of_conditions
            @test eval_dual(constraint) isa Float64
        end

        @test length([eval_dual(constraint) for constraint in func._PEPit_func.list_of_constraints]) == 2
        for constraint in func._PEPit_func.list_of_constraints
            @test eval_dual(constraint) ≈ 1.8 atol = 1e-4
        end
    end
end

@testset "Constraints: logdet heuristic" begin

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
        f = declare_function!(p, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu))
        xs = stationary_point!(f)
        x0 = set_initial_point!(p)
        set_initial_condition!(p, (x0 - xs)^2 <= 1)
        x1 = x0 - gamma * gradient!(f, x0)
        set_performance_metric!(p, (x1 - xs)^2)
        p
    end


    pb_base = build_problem()
    τ_base = solve!(pb_base; verbose=false)


    pb_logdet = build_problem()
    τ_logdet = solve!(pb_logdet; verbose=false, logdetiters=10, eig_regularization=1e-3, tol_dimension_reduction=1e-5)

    @test isapprox(τ_logdet, τ_base; atol=1e-6, rtol=1e-3)
end
