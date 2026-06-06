mutable struct LipschitzStronglyMonotoneOperatorCheap <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function LipschitzStronglyMonotoneOperatorCheap(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        L == Inf && @warn "(PEPit) L == Inf; consider StronglyMonotoneOperator instead."
        return new(mu, L, func)
    end
end

add_constraint!(op::LipschitzStronglyMonotoneOperatorCheap, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::LipschitzStronglyMonotoneOperatorCheap) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::LipschitzStronglyMonotoneOperatorCheap)
    pts = op._PEPit_func.list_of_points

    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) - op.mu * (xi - xj)^2 >= 0)
    end

    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj)^2 - op.L^2 * (xi - xj)^2 <= 0)
    end
end

_get_pep_func(op::LipschitzStronglyMonotoneOperatorCheap) = op._PEPit_func
