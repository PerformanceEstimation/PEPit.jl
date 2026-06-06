using OrderedCollections

mutable struct ConvexIndicatorFunction <: AbstractFunction
    D::Float64
    R::Float64
    center::Union{Point,Nothing}
    _PEPit_func::PEPFunction


    function ConvexIndicatorFunction(param=OrderedDict();
        is_leaf::Bool=true,
        decomposition_dict=nothing,
        reuse_gradient::Bool=false)
        @assert is_leaf

        D = haskey(param, "D") ? float(param["D"]) : Inf
        R = haskey(param, "R") ? float(param["R"]) : Inf
        c = haskey(param, "center") ? param["center"] : nothing


        if c === nothing && R != Inf
            c = Point()
        end

        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(D, R, c, func)
    end


end


gradient!(f::ConvexIndicatorFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexIndicatorFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexIndicatorFunction) = stationary_point!(f._PEPit_func)
add_constraint!(f::ConvexIndicatorFunction, c::Constraint) = add_constraint!(f._PEPit_func, c)


function add_class_constraints!(f::ConvexIndicatorFunction)
    points_list = f._PEPit_func.list_of_points


    for point_i in points_list
        xi, gi, fi = point_i
        add_constraint!(f, fi == 0)
    end


    for point_i in points_list, point_j in points_list
        if point_i === point_j
            continue
        end
        xi, gi, fi = point_i
        xj, gj, fj = point_j
        add_constraint!(f, 0 >= gj * (xi - xj))
    end


    if f.D != Inf
        D2 = f.D^2
        for point_i in points_list, point_j in points_list
            if point_i === point_j
                continue
            end
            xi, gi, fi = point_i
            xj, gj, fj = point_j
            add_constraint!(f, (xi - xj)^2 <= D2)
        end
    end


    if f.R != Inf
        @assert f.center isa Point
        R2 = f.R^2
        for point_i in points_list
            xi, gi, fi = point_i
            add_constraint!(f, (xi - (f.center::Point))^2 <= R2)
        end
    end


end

_get_pep_func(f::ConvexIndicatorFunction) = f._PEPit_func
