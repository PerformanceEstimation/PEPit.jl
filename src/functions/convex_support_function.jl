mutable struct ConvexSupportFunction <: AbstractFunction
    M::Float64
    _PEPit_func::PEPFunction

    function ConvexSupportFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        M = haskey(param, "M") ? float(param["M"]) : Inf
        return new(M, func)
    end
end

gradient!(f::ConvexSupportFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexSupportFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexSupportFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexSupportFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexSupportFunction)
    points = func._PEPit_func.list_of_points


    for (xi, gi, fi) in points
        add_constraint!(func, gi * xi - fi == 0)
    end


    if func.M != Inf
        M2 = func.M^2
        for (xi, gi, fi) in points
            add_constraint!(func, gi^2 <= M2)
        end
    end


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, xj * (gi - gj) <= 0)
    end
end

_get_pep_func(f::ConvexSupportFunction) = f._PEPit_func
