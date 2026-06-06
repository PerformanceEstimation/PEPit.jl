using Test

@testset "OrderedDict operations" begin

    dict1 = OrderedDict("a" => 5, "b" => 6, "q" => 11)
    dict2 = OrderedDict("a" => 2, "b" => 8, "w" => 0)


    summed_dict = OrderedDict("a" => 7, "b" => 14, "q" => 11, "w" => 0)
    @test merge_dicts(dict1, dict2) == summed_dict


    product_dict = OrderedDict(
        ("a", "a") => 10, ("a", "b") => 40, ("a", "w") => 0,
        ("b", "a") => 12, ("b", "b") => 48, ("b", "w") => 0,
        ("q", "a") => 22, ("q", "b") => 88, ("q", "w") => 0,
    )
    @test multiply_dicts(dict1, dict2) == product_dict


    @test prune_dict(dict2) == OrderedDict("a" => 2, "b" => 8)
end
