using PEPit
using Test
using OrderedCollections
using JuMP, Mosek, MosekTools

@info "💻 Running PEPit.jl tests!"

ts = @testset verbose=true "PEPit.jl" begin
    include("test_constraint.jl")
    include("test_dict_operations.jl")
    include("test_dualization.jl")
    include("test_expression.jl")
    include("test_function.jl")
    include("test_point.jl")
    include("test_pep.jl")
    include("test_additions.jl")
end

if !ts.anynonpass
    @info "🎉 All tests passed!"
else
    @error "💀 Some tests failed!"
end
