@doc raw"""
    BlockSmoothConvexFunctionCheap(param; <keyword arguments>)

Represent the `BlockSmoothConvexFunctionCheap` interpolation class in PEPit.jl.

Implement necessary constraints for interpolation of the class of smooth convex functions by blocks.

# Warning

    Functions that are smooth by blocks and convex generally do not enjoy known interpolation conditions.
    The conditions implemented in this class are necessary but a priori not sufficient for interpolation.
    Hence, the numerical results obtained when using this class might be non-tight upper bounds.

# Class parameters
- `partition`: partitioning of the variables (in blocks).
- `L`: smoothness parameters (one per block).

Smooth convex functions by blocks are characterized by a list of parameters $L_i$ (one per block),
hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, BlockSmoothConvexFunctionCheap, param)
```

# Fields
- `partition`: class parameter or auxiliary state stored as `BlockPartition`.
- `L`: class parameter or auxiliary state stored as `Vector{Float64}`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct BlockSmoothConvexFunctionCheap <: AbstractFunction
    partition::BlockPartition
    L::Vector{Float64}
    _PEPit_func::PEPFunction

    function BlockSmoothConvexFunctionCheap(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        partition = param["partition"]
        Lraw = param["L"]
        L = Lraw isa AbstractVector ? Float64.(Lraw) : [Float64(Lraw)]
        if get_nb_blocks(partition) > 1
            @assert length(L) == get_nb_blocks(partition) "length(L) must equal the number of blocks."
            @assert all(isfinite, L)
        end
        return new(partition, L, func)
    end
end

gradient!(f::BlockSmoothConvexFunctionCheap, p::Point) = gradient!(f._PEPit_func, p)
value!(f::BlockSmoothConvexFunctionCheap, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::BlockSmoothConvexFunctionCheap) = stationary_point!(f._PEPit_func)
add_constraint!(func::BlockSmoothConvexFunctionCheap, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::BlockSmoothConvexFunctionCheap)
    points = func._PEPit_func.list_of_points
    nb = get_nb_blocks(func.partition)
    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        for k in 1:nb
            gik = get_block(func.partition, gi, k)
            gjk = get_block(func.partition, gj, k)
            add_constraint!(func, fi - fj >= gj * (xi - xj) + 1 / (2 * func.L[k]) * (gik - gjk)^2)
        end
    end
end

_get_pep_func(f::BlockSmoothConvexFunctionCheap) = f._PEPit_func
