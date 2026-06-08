@doc raw"""
    BlockSmoothConvexFunctionExpensive(param; <keyword arguments>)

Represent the `BlockSmoothConvexFunctionExpensive` interpolation class in PEPit.jl.

The `RefinedBlockSmoothConvexFunctionExpensive` class overwrites the `add_class_constraints` method
of [`PEPFunction`](@ref), by implementing necessary constraints for interpolation of the class of
smooth convex functions by blocks. The implemented constraint is that of [2, Section 3.1].

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
f = declare_function!(problem, BlockSmoothConvexFunctionExpensive, param)
```

# Fields
- `partition`: class parameter or auxiliary state stored as `BlockPartition`.
- `L`: class parameter or auxiliary state stored as `Vector{Float64}`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct BlockSmoothConvexFunctionExpensive <: AbstractFunction
    partition::BlockPartition
    L::Vector{Float64}
    _PEPit_func::PEPFunction

    function BlockSmoothConvexFunctionExpensive(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
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

gradient!(f::BlockSmoothConvexFunctionExpensive, p::Point) = gradient!(f._PEPit_func, p)
value!(f::BlockSmoothConvexFunctionExpensive, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::BlockSmoothConvexFunctionExpensive) = stationary_point!(f._PEPit_func)
add_constraint!(func::BlockSmoothConvexFunctionExpensive, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::BlockSmoothConvexFunctionExpensive)
    internal = func._PEPit_func
    points = internal.list_of_points
    nb = get_nb_blocks(func.partition)
    n = length(points)
    for i in 1:n, j in 1:n, k in 1:n
        (i == j && i == k) && continue
        xi, gi, fi = points[i]
        xj, gj, fj = points[j]
        xk, gk, fk = points[k]
        for m in 1:nb, l in 1:nb
            gim = get_block(func.partition, gi, m)
            gjm = get_block(func.partition, gj, m)
            gkm = get_block(func.partition, gk, m)
            gjl = get_block(func.partition, gj, l)
            gkl = get_block(func.partition, gk, l)

            new_expr = Expression()
            add_constraint!(func, new_expr >= 0)

            A = -(-fi + fk + gk * (xi - xk) + 1 / (2 * func.L[m]) * (gim - gkm)^2)
            B = -(-fi + fj + gj * (xi - xj) + 1 / (2 * func.L[m]) * (gim - gjm)^2)
            C = -(1 / (2 * func.L[l]) * (gjl - gkl)^2 - 1 / (2 * func.L[m]) * (gjm - gkm)^2)

            off = 0.5 * (A + B + C) - new_expr
            T = Matrix{Expression}(undef, 2, 2)
            T[1, 1] = A; T[1, 2] = off; T[2, 1] = off; T[2, 2] = B
            push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T))
        end
    end
end

_get_pep_func(f::BlockSmoothConvexFunctionExpensive) = f._PEPit_func
