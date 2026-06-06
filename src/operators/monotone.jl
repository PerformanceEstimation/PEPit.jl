mutable struct MonotoneOperator <: AbstractFunction
    _PEPit_func::PEPFunction

    function MonotoneOperator(param=OrderedDict(); is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(func)
    end
end

add_constraint!(op::MonotoneOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)

function add_class_constraints!(op::MonotoneOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) >= 0)
    end
end

_get_pep_func(op::MonotoneOperator) = op._PEPit_func
