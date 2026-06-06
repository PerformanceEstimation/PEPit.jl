mutable struct ConvexQGFunction <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function ConvexQGFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["L"]), func)
    end
end

gradient!(f::ConvexQGFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexQGFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexQGFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexQGFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexQGFunction)
    internal = func._PEPit_func
    if isempty(internal.list_of_stationary_points)
        stationary_point!(internal)
    end
    points = internal.list_of_points
    stationary = internal.list_of_stationary_points


    for (xi, gi, fi) in stationary, (xj, gj, fj) in points
        xi == xj && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj) + 1 / (2 * func.L) * gj^2)
    end


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj))
    end
end

_get_pep_func(f::ConvexQGFunction) = f._PEPit_func
