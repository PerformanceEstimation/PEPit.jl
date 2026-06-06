mutable struct SkewSymmetricLinearOperator <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function SkewSymmetricLinearOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["L"]), func)
    end
end

add_constraint!(op::SkewSymmetricLinearOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::SkewSymmetricLinearOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::SkewSymmetricLinearOperator)
    internal = op._PEPit_func
    pts = internal.list_of_points


    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, xi * gj == -(xj * gi))
    end


    for (xi, gi, _) in pts
        add_constraint!(op, xi * gi == 0)
    end


    N = length(pts)
    if N > 0
        T = Matrix{Expression}(undef, N, N)
        for (i, (xi, gi, fi)) in enumerate(pts), (j, (xj, gj, fj)) in enumerate(pts)
            T[i, j] = (op.L^2) * (xi * xj) - gi * gj
        end
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T))
    end
end

_get_pep_func(op::SkewSymmetricLinearOperator) = op._PEPit_func
