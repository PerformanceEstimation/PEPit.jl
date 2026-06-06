mutable struct SmoothConvexLipschitzFunction <: AbstractFunction
    L::Float64
    M::Float64
    _PEPit_func::PEPFunction

    function SmoothConvexLipschitzFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["L"]), float(param["M"]), func)
    end
end

gradient!(f::SmoothConvexLipschitzFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothConvexLipschitzFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothConvexLipschitzFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothConvexLipschitzFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothConvexLipschitzFunction)
    points = func._PEPit_func.list_of_points


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj) + 1 / (2 * func.L) * (gi - gj)^2)
    end


    M2 = func.M^2
    for (xi, gi, fi) in points
        add_constraint!(func, gi^2 <= M2)
    end
end

_get_pep_func(f::SmoothConvexLipschitzFunction) = f._PEPit_func
