mutable struct ConvexLipschitzFunction <: AbstractFunction
    M::Float64
    _PEPit_func::PEPFunction

    function ConvexLipschitzFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        M = float(param["M"])
        if M == Inf
            @warn "(PEPit) The class of convex M-Lipschitz functions with M == Inf implies no constraint: it contains all convex closed proper functions."
        end
        return new(M, func)
    end
end


gradient!(f::ConvexLipschitzFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexLipschitzFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexLipschitzFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexLipschitzFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexLipschitzFunction)
    points_list = func._PEPit_func.list_of_points


    if func.M != Inf
        M2 = func.M^2
        for point_i in points_list
            _, gi, _ = point_i
            add_constraint!(func, gi^2 <= M2)
        end
    end


    for point_i in points_list, point_j in points_list
        if point_i == point_j
            continue
        end
        xi, gi, fi = point_i
        xj, gj, fj = point_j
        add_constraint!(func, fi - fj >= gj * (xi - xj))
    end
end

_get_pep_func(f::ConvexLipschitzFunction) = f._PEPit_func
