mutable struct StronglyMonotoneOperator <: AbstractFunction
    mu::Float64
    _PEPit_func::PEPFunction

    function StronglyMonotoneOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["mu"]), func)
    end
end

add_constraint!(op::StronglyMonotoneOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::StronglyMonotoneOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::StronglyMonotoneOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) - op.mu * (xi - xj)^2 >= 0)
    end
end

_get_pep_func(op::StronglyMonotoneOperator) = op._PEPit_func
