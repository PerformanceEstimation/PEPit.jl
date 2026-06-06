using Test


function reset_counters!()
    Point_counter[] = 0
    Expression_counter[] = 0
    Function_counter[] = 0
    Global_Constraint_counter[] = 0
    NEXT_ID[] = 0
end

@testset "PEPFunction Tests" begin


    function setup_test()
        reset_counters!()
        pep = PEP()
        func1 = PEPFunction(is_leaf=true, decomposition_dict=nothing)
        func2 = ConvexFunction(OrderedDict(); is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        point = Point(is_leaf=true, decomposition_dict=nothing)
        return pep, func1, func2, point
    end

    @testset "Instance Type Tests" begin
        pep, func1, func2, point = setup_test()

        @test func1 isa PEPFunction
        @test func2 isa ConvexFunction
        @test func2._PEPit_func isa PEPFunction
    end

    @testset "Counter Tests" begin
        pep, func1, func2, point = setup_test()

        composite_function = func1 + func2
        @test func1.counter == 0
        @test func2._PEPit_func.counter == 1
        @test composite_function.counter === nothing
        @test Function_counter[] == 2

        new_function = PEPFunction(is_leaf=true, decomposition_dict=nothing)
        @test new_function.counter == 2
        @test Function_counter[] == 3
    end

    function compute_linear_combination(func1, func2)
        new_function = -1 * func1 + 2 * func2 - func2 / 5
        return new_function
    end

    @testset "Linear Combination Tests" begin
        pep, func1, func2, point = setup_test()

        new_function = compute_linear_combination(func1, func2)

        @test new_function isa PEPFunction
        expected_dict = OrderedDict(func1 => -1.0, func2._PEPit_func => 9.0 / 5.0)
        @test new_function.decomposition_dict == expected_dict
    end

    @testset "Oracle Tests" begin
        pep, func1, func2, point = setup_test()

        new_function = compute_linear_combination(func1, func2)
        oracle!(new_function, point)


        @test length(new_function.list_of_points) == 1

        point_result, grad, val = new_function.list_of_points[1]
        @test point_result isa Point
        @test grad isa Point
        @test val isa Expression

        @test point_result === point
        @test get_is_leaf(grad)
        @test get_is_leaf(val)


        @test length(func1.list_of_points) == 1

        point1, grad1, val1 = func1.list_of_points[1]
        @test point1 isa Point
        @test grad1 isa Point
        @test val1 isa Expression

        @test get_is_leaf(grad1)
        @test get_is_leaf(val1)


        @test length(func2._PEPit_func.list_of_points) == 1

        point2, grad2, val2 = func2._PEPit_func.list_of_points[1]
        @test point2 isa Point
        @test grad2 isa Point
        @test val2 isa Expression

        @test !get_is_leaf(grad2)
        @test !get_is_leaf(val2)


        @test point1 === point
        @test point2 === point

        expected_grad = -1 * grad1 + 9 / 5 * grad2
        expected_val = -1 * val1 + 9 / 5 * val2
        @test prune_dict(expected_grad.decomposition_dict) == grad.decomposition_dict
        @test prune_dict(expected_val.decomposition_dict) == val.decomposition_dict
    end

    @testset "Oracle with Predetermined Values" begin
        pep, func1, func2, point = setup_test()


        new_function = compute_linear_combination(func1, func2)


        grad1, val1 = oracle!(func1, point)
        grad2, val2 = oracle!(func2._PEPit_func, point)


        @test length(func1.list_of_points) == 1
        @test length(func2._PEPit_func.list_of_points) == 1


        grad, val = oracle!(new_function, point)


        expected_val_dict = prune_dict((-1 * val1 + 9 / 5 * val2).decomposition_dict)
        expected_grad_dict = prune_dict((-1 * grad1 + 9 / 5 * grad2).decomposition_dict)
        @test prune_dict(val.decomposition_dict) == expected_val_dict
        @test prune_dict(grad.decomposition_dict) != expected_grad_dict


        @test length(func1.list_of_points) == 2
        @test length(func2._PEPit_func.list_of_points) == 2


        other_grad1, other_val1 = func1.list_of_points[2][2:3]
        other_grad2, other_val2 = func2._PEPit_func.list_of_points[2][2:3]


        @test val1.decomposition_dict == other_val1.decomposition_dict
        @test val2.decomposition_dict == other_val2.decomposition_dict


        expected_new_grad_dict = prune_dict((-1 * other_grad1 + 9 / 5 * other_grad2).decomposition_dict)
        @test prune_dict(grad.decomposition_dict) == expected_new_grad_dict
    end

    @testset "Oracle with Predetermined Values and Gradients" begin
        pep, func1, func2, point = setup_test()


        func1.reuse_gradient = true
        func2._PEPit_func.reuse_gradient = true


        new_function = compute_linear_combination(func1, func2)


        @test new_function.reuse_gradient


        grad1, val1 = oracle!(func1, point)
        grad2, val2 = oracle!(func2._PEPit_func, point)


        @test length(func1.list_of_points) == 1
        @test length(func2._PEPit_func.list_of_points) == 1


        grad, val = oracle!(new_function, point)


        expected_val_dict = prune_dict((-1 * val1 + 9 / 5 * val2).decomposition_dict)
        expected_grad_dict = prune_dict((-1 * grad1 + 9 / 5 * grad2).decomposition_dict)
        @test prune_dict(val.decomposition_dict) == expected_val_dict
        @test prune_dict(grad.decomposition_dict) == expected_grad_dict


        @test length(func1.list_of_points) == 1
        @test length(func2._PEPit_func.list_of_points) == 1
    end

    @testset "Stationary Point Tests" begin
        pep, func1, func2, point = setup_test()


        new_function = compute_linear_combination(func1, func2)
        stationary_point!(new_function)


        @test length(new_function.list_of_points) == 1

        point_result, grad, val = new_function.list_of_points[1]
        @test point_result isa Point
        @test grad isa Point
        @test val isa Expression

        @test get_is_leaf(point_result)
        @test !get_is_leaf(grad)
        @test get_is_leaf(val)

        @test isempty(grad.decomposition_dict)
        @test isempty((grad^2).decomposition_dict)

        @test length(new_function.list_of_stationary_points) == 1


        @test length(func1.list_of_points) == 1

        point1, grad1, val1 = func1.list_of_points[1]
        @test point1 isa Point
        @test grad1 isa Point
        @test val1 isa Expression

        @test get_is_leaf(grad1)
        @test get_is_leaf(val1)

        @test length(func1.list_of_stationary_points) == 0


        @test length(func2._PEPit_func.list_of_points) == 1

        point2, grad2, val2 = func2._PEPit_func.list_of_points[1]
        @test point2 isa Point
        @test grad2 isa Point
        @test val2 isa Expression

        @test !get_is_leaf(grad2)
        @test !get_is_leaf(val2)

        @test length(func2._PEPit_func.list_of_stationary_points) == 0


        @test point1 === point_result
        @test point2 === point_result

        expected_grad_dict = (-1 * grad1 + 9 / 5 * grad2).decomposition_dict
        expected_val_dict = prune_dict((-1 * val1 + 9 / 5 * val2).decomposition_dict)
        @test expected_grad_dict == grad.decomposition_dict
        @test expected_val_dict == val.decomposition_dict
    end

    @testset "Is Already Evaluated on Points" begin
        pep, func1, func2, point = setup_test()

        new_function = compute_linear_combination(func1, func2)


        @test _is_already_evaluated_on_point(new_function, point) === nothing
        @test _is_already_evaluated_on_point(func1, point) === nothing
        @test _is_already_evaluated_on_point(func2._PEPit_func, point) === nothing


        oracle!(new_function, point)


        @test _is_already_evaluated_on_point(new_function, point) == new_function.list_of_points[1][2:3]
        @test _is_already_evaluated_on_point(func1, point) == func1.list_of_points[1][2:3]
        @test _is_already_evaluated_on_point(func2._PEPit_func, point) == func2._PEPit_func.list_of_points[1][2:3]
    end

    @testset "Separate Leaf Functions - Non-differentiable" begin
        pep, func1, func2, point = setup_test()


        new_function = compute_linear_combination(func1, func2)
        point1 = Point(is_leaf=true, decomposition_dict=nothing)
        point2 = Point(is_leaf=true, decomposition_dict=nothing)
        oracle!(new_function, point1)

        list_nothing, list_grad_only, list_grad_val = _separate_leaf_functions_regarding_their_need_on_point(new_function, point1)
        @test length(list_nothing) == 0
        @test length(list_grad_only) == 2
        @test length(list_grad_val) == 0

        list_nothing, list_grad_only, list_grad_val = _separate_leaf_functions_regarding_their_need_on_point(new_function, point2)
        @test length(list_nothing) == 0
        @test length(list_grad_only) == 0
        @test length(list_grad_val) == 2
    end

    @testset "Separate Leaf Functions - Differentiable" begin
        reset_counters!()


        new_function = PEPFunction(is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        point1 = Point(is_leaf=true, decomposition_dict=nothing)
        point2 = Point(is_leaf=true, decomposition_dict=nothing)
        oracle!(new_function, point1)

        list_nothing, list_grad_only, list_grad_val = _separate_leaf_functions_regarding_their_need_on_point(new_function, point1)
        @test length(list_nothing) == 1
        @test length(list_grad_only) == 0
        @test length(list_grad_val) == 0

        list_nothing, list_grad_only, list_grad_val = _separate_leaf_functions_regarding_their_need_on_point(new_function, point2)
        @test length(list_nothing) == 0
        @test length(list_grad_only) == 0
        @test length(list_grad_val) == 1
    end

end
