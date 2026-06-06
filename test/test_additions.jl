function _gd1_value(f, problem, xs, fs; step=0.1, solver=Mosek.Optimizer)
    x0 = set_initial_point!(problem)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1.0)
    g0 = gradient!(f, x0)
    x1 = x0 - step * g0
    set_performance_metric!(problem, value!(f, x1) - fs)
    return solve!(problem; solver=solver, verbose=false)
end


function _op1_dist(A, problem; step=0.3, solver=Mosek.Optimizer)
    xs = stationary_point!(A)
    x0 = set_initial_point!(problem)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1.0)
    g0 = gradient!(A, x0)
    x1 = x0 - step * g0
    set_performance_metric!(problem, (x1 - xs)^2)
    return solve!(problem; solver=solver, verbose=false)
end

@testset "BlockPartition" begin
    problem = PEP()
    bp = declare_block_partition!(problem, 3)
    @test bp isa BlockPartition
    @test get_nb_blocks(bp) == 3

    x = Point()
    blocks = [get_block(bp, x, k) for k in 1:3]

    s = blocks[1] + blocks[2] + blocks[3]
    @test isempty((x - s).decomposition_dict)

    @test get_block(bp, x, 2) === blocks[2]


    y = Point()
    for k in 1:3
        get_block(bp, y, k)
    end
    PEPit.add_partition_constraints!(bp)
    @test length(bp.list_of_constraints) == 12


    problem2 = PEP()
    bp1 = declare_block_partition!(problem2, 1)
    z = Point()
    @test isempty((get_block(bp1, z, 1) - z).decomposition_dict)


    _ = PEP()
    @test isempty(PEPit.GLOBAL_BLOCK_PARTITIONS)
    @test PEPit.BlockPartition_counter[] == 0
end

@testset "New function classes (construct + solve)" begin

    function_cases = [
        ("ConvexQGFunction", () -> begin p = PEP(); f = declare_function!(p, ConvexQGFunction, OrderedDict("L" => 1.0)); (p, f) end),
        ("ConvexSupportFunction", () -> begin p = PEP(); f = declare_function!(p, ConvexSupportFunction, OrderedDict("M" => 1.0)); (p, f) end),
        ("RsiEbFunction", () -> begin p = PEP(); f = declare_function!(p, RsiEbFunction, OrderedDict("mu" => 0.1, "L" => 1.0)); (p, f) end),
        ("SmoothConvexLipschitzFunction", () -> begin p = PEP(); f = declare_function!(p, SmoothConvexLipschitzFunction, OrderedDict("L" => 1.0, "M" => 1.0)); (p, f) end),
        ("SmoothStronglyConvexQuadraticFunction", () -> begin p = PEP(); f = declare_function!(p, SmoothStronglyConvexQuadraticFunction, OrderedDict("mu" => 0.1, "L" => 1.0)); (p, f) end),
        ("SmoothQuadraticLojasiewiczFunctionCheap", () -> begin p = PEP(); f = declare_function!(p, SmoothQuadraticLojasiewiczFunctionCheap, OrderedDict("L" => 1.0, "mu" => 0.5, "alpha" => 0.4)); (p, f) end),
        ("SmoothQuadraticLojasiewiczFunctionExpensive", () -> begin p = PEP(); f = declare_function!(p, SmoothQuadraticLojasiewiczFunctionExpensive, OrderedDict("L" => 1.0, "mu" => 0.1)); (p, f) end),
    ]
    for (name, build) in function_cases
        p, f = build()
        xs = stationary_point!(f)
        fs = value!(f, xs)
        tau = _gd1_value(f, p, xs, fs)
        @test isfinite(tau)
        @test tau > 0
    end


    for Cls in (BlockSmoothConvexFunctionCheap, BlockSmoothConvexFunctionExpensive)
        p = PEP()
        part = declare_block_partition!(p, 2)
        f = declare_function!(p, Cls, OrderedDict("partition" => part, "L" => [1.0, 4.0]))
        xs = stationary_point!(f)
        fs = value!(f, xs)
        tau = _gd1_value(f, p, xs, fs)
        @test isfinite(tau)
        @test tau > 0
    end
end

@testset "New operators (construct + solve)" begin
    operator_cases = [
        ("StronglyMonotoneOperator", () -> begin p = PEP(); A = declare_function!(p, StronglyMonotoneOperator, OrderedDict("mu" => 0.1)); (p, A) end),
        ("NegativelyComonotoneOperator", () -> begin p = PEP(); A = declare_function!(p, NegativelyComonotoneOperator, OrderedDict("rho" => 0.1)); (p, A) end),
        ("CocoerciveOperator", () -> begin p = PEP(); A = declare_function!(p, CocoerciveOperator, OrderedDict("beta" => 1.0)); (p, A) end),
        ("CocoerciveStronglyMonotoneOperatorCheap", () -> begin p = PEP(); A = declare_function!(p, CocoerciveStronglyMonotoneOperatorCheap, OrderedDict("mu" => 0.1, "beta" => 1.0)); (p, A) end),
        ("CocoerciveStronglyMonotoneOperatorExpensive", () -> begin p = PEP(); A = declare_function!(p, CocoerciveStronglyMonotoneOperatorExpensive, OrderedDict("mu" => 0.1, "beta" => 1.0)); (p, A) end),
        ("LipschitzStronglyMonotoneOperatorCheap", () -> begin p = PEP(); A = declare_function!(p, LipschitzStronglyMonotoneOperatorCheap, OrderedDict("mu" => 0.1, "L" => 1.0)); (p, A) end),
        ("LipschitzStronglyMonotoneOperatorExpensive", () -> begin p = PEP(); A = declare_function!(p, LipschitzStronglyMonotoneOperatorExpensive, OrderedDict("mu" => 0.1, "L" => 1.0)); (p, A) end),
        ("SymmetricLinearOperator", () -> begin p = PEP(); A = declare_function!(p, SymmetricLinearOperator, OrderedDict("mu" => 0.1, "L" => 1.0)); (p, A) end),
        ("SkewSymmetricLinearOperator", () -> begin p = PEP(); A = declare_function!(p, SkewSymmetricLinearOperator, OrderedDict("L" => 1.0)); (p, A) end),
    ]
    for (name, build) in operator_cases
        p, A = build()
        tau = _op1_dist(A, p)
        @test isfinite(tau)
        @test tau > 0
    end
end
