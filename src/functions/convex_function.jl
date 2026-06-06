mutable struct ConvexFunction <: AbstractFunction
    _PEPit_func::PEPFunction

    function ConvexFunction(param=OrderedDict(); is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(func)
    end
end


gradient!(f::ConvexFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexFunction)
    points_list = func._PEPit_func.list_of_points
    for point_i in points_list, point_j in points_list
        if point_i == point_j
            continue
        end
        xi, gi, fi = point_i
        xj, gj, fj = point_j
        constraint = (fi - fj >= gj * (xi - xj))
        add_constraint!(func, constraint)
    end
end

_get_pep_func(f::ConvexFunction) = f._PEPit_func
