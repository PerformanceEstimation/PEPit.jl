using Test

@testset "Point" begin

    Point_counter[] = 0
    Expression_counter[] = 0
    Global_Constraint_counter[] = 0
    NEXT_ID[] = 0

    pep = PEP()
    A = Point()
    B = Point()


    @test A isa Point
    @test B isa Point


    C = A + B
    @test A.counter == 0
    @test B.counter == 1
    @test C.counter === nothing
    @test Point_counter[] == 2

    D = Point()
    @test D.counter == 2
    @test Point_counter[] == 3


    new_point = (-A) * 1.0 + 2 * B - (B / 5)
    @test new_point isa Point
    @test new_point.decomposition_dict == OrderedDict(A => -1.0, B => 9.0 / 5.0)


    inner_product = A * B
    @test inner_product isa Expression
    @test !inner_product._is_leaf
    @test inner_product.decomposition_dict == OrderedDict((A, B) => 1.0)


    norm_square = (A - B)^2
    @test norm_square isa Expression
    @test !norm_square._is_leaf
    @test norm_square.decomposition_dict == OrderedDict(
        (A, A) => 1.0,
        (A, B) => -1.0,
        (B, A) => -1.0,
        (B, B) => 1.0,
    )
end
