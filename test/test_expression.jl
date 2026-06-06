using Test

@testset "Expression" begin

    Point_counter[] = 0
    Expression_counter[] = 0
    Global_Constraint_counter[] = 0
    NEXT_ID[] = 0


    pep = PEP()
    point1 = Point()
    point2 = Point()
    inner_product = point1 * point2
    function_value = Expression()


    @test inner_product isa Expression
    @test function_value isa Expression


    composite_expression = inner_product + function_value
    @test composite_expression.counter === nothing
    @test inner_product.counter === nothing
    @test function_value.counter == 0
    @test Expression_counter[] == 1

    new_expression = Expression()
    @test new_expression.counter == 1
    @test Expression_counter[] == 2


    new_expression2 = 1 + 2 * (4 - (-(inner_product) * 3) - 5 + 2 * function_value - function_value / 5 + 2)
    @test new_expression2 isa Expression
    @test new_expression2.decomposition_dict == OrderedDict(
        1 => 3.0,
        (point1, point2) => 6.0,
        function_value => 18.0 / 5.0,
    )


    constraint = inner_product <= function_value
    @test constraint isa Constraint
    @test constraint.expression.decomposition_dict == (inner_product - function_value).decomposition_dict
end
