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
