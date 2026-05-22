using Test
using LinearAlgebra

function _dual_test_gd_problem()
    problem = PEP()
    param = OrderedDict("mu" => 0.1, "L" => 1.0)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)
    xs = stationary_point!(func)
    x0 = set_initial_point!(problem)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)
    x1 = x0 - gradient!(func, x0)
    set_performance_metric!(problem, (x1 - xs)^2)
    return problem
end

function _dual_test_equality_problem()
    problem = PEP()
    x = set_initial_point!(problem)
    set_initial_condition!(problem, x^2 == 1)
    set_performance_metric!(problem, x^2)
    return problem
end

function _dual_test_psd_problem()
    problem = PEP()
    x = set_initial_point!(problem)
    expr = Expression()
    set_initial_condition!(problem, x^2 <= 9.0)
    psd = add_psd_matrix!(problem, [[x^2, expr], [expr, 1]])
    set_performance_metric!(problem, expr)
    return problem, psd
end

@testset "Explicit Dualization" begin
    @testset "Gradient descent dual certificate" begin
        primal_problem = _dual_test_gd_problem()
        dual_problem = _dual_test_gd_problem()

        primal_value = solve!(primal_problem; verbose=false)
        certificate = solve_dual!(dual_problem; verbose=false)

        @test certificate isa DualPEPCertificate
        @test isapprox(certificate.dual_value, primal_value; atol=1e-5, rtol=1e-5)
        @test length(certificate.α) == 1
        @test isapprox(sum(certificate.α), 1.0; atol=1e-7)
        @test all(certificate.α .>= -1e-7)
        @test length(certificate.λ) == 1
        @test length(certificate.ν) == 0
        @test all(certificate.λ .>= -1e-7)
        @test isapprox(eval_dual(dual_problem.list_of_conditions[1]), certificate.λ[1]; atol=1e-8)
        @test minimum(eigvals(Symmetric(certificate.S))) >= -1e-6
        @test dual_problem.residual == certificate.S
    end

    @testset "Equality multipliers use ν" begin
        primal_problem = _dual_test_equality_problem()
        dual_problem = _dual_test_equality_problem()

        primal_value = solve!(primal_problem; verbose=false)
        certificate = solve_dual!(dual_problem; verbose=false)

        @test isapprox(certificate.dual_value, primal_value; atol=1e-5, rtol=1e-5)
        @test length(certificate.λ) == 0
        @test length(certificate.ν) == 1
        @test isapprox(eval_dual(dual_problem.list_of_conditions[1]), certificate.ν[1]; atol=1e-8)
    end

    @testset "Problem-level PSD dual cache" begin
        primal_problem, _ = _dual_test_psd_problem()
        dual_problem, psd = _dual_test_psd_problem()

        primal_value = solve!(primal_problem; verbose=false)
        certificate = solve_dual!(dual_problem; verbose=false)

        @test isapprox(certificate.dual_value, primal_value; atol=1e-5, rtol=1e-5)
        @test length(certificate.Y.problem) == 1
        @test size(certificate.Y.problem[1]) == (2, 2)
        @test isapprox(eval_dual(psd), certificate.Y.problem[1]; atol=1e-8)
        @test minimum(eigvals(Symmetric(certificate.Y.problem[1]))) >= -1e-6
    end
end
