mutable struct NegativelyComonotoneOperator <: AbstractFunction
    rho::Float64
    _PEPit_func::PEPFunction

    function NegativelyComonotoneOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        rho = float(param["rho"])
        rho < 0 && @warn "(PEPit) The parameter rho is expected to be positive."
        rho == 0 && @warn "(PEPit) rho == 0 reduces to a monotone operator; consider MonotoneOperator instead."
        return new(rho, func)
    end
end

add_constraint!(op::NegativelyComonotoneOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::NegativelyComonotoneOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::NegativelyComonotoneOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) + op.rho * (gi - gj)^2 >= 0)
    end
end

_get_pep_func(op::NegativelyComonotoneOperator) = op._PEPit_func
