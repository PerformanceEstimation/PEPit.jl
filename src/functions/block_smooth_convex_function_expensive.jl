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
