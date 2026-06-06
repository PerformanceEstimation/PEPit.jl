mutable struct PEPFunction <: AbstractFunction
    _id::Int
    _is_leaf::Bool
    decomposition_dict::OrderedDict{PEPFunction,Float64}
    reuse_gradient::Bool
    list_of_points::Vector{Tuple{Point,Point,Expression}}
    list_of_stationary_points::Vector{Tuple{Point,Point,Expression}}
    list_of_constraints::Vector{Constraint}
    list_of_psd::Vector{PSDMatrix}
    list_of_class_psd::Vector{PSDMatrix}
    counter::Union{Int,Nothing}

    function PEPFunction(; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        id = (NEXT_ID[] += 1)
        if is_leaf
            @assert decomposition_dict === nothing
            func = new(id, true,
                OrderedDict{PEPFunction,Float64}(),
                reuse_gradient,
                [], [], [],
                [], [],
                Function_counter[])
            func.decomposition_dict[func] = 1.0
            Function_counter[] += 1
            return func
        else
            @assert decomposition_dict isa OrderedDict
            return new(id, false,
                decomposition_dict,
                reuse_gradient,
                [], [], [],
                [], [],
                nothing)
        end
    end
end


Base.hash(f::PEPFunction, h::UInt) = hash(f._id, h)
Base.:(==)(f1::PEPFunction, f2::PEPFunction) = f1._id == f2._id
Base.isequal(f1::PEPFunction, f2::PEPFunction) = f1._id == f2._id


_get_pep_func(f::PEPFunction) = f


function +(f1::AbstractFunction, f2::AbstractFunction)
    pf1 = _get_pep_func(f1)
    pf2 = _get_pep_func(f2)
    merged_decomposition_dict = merge_dicts(pf1.decomposition_dict, pf2.decomposition_dict)
    return PEPFunction(is_leaf=false,
        decomposition_dict=merged_decomposition_dict,
        reuse_gradient=pf1.reuse_gradient && pf2.reuse_gradient)
end

function *(s::Real, f::AbstractFunction)
    pf = _get_pep_func(f)
    if pf._is_leaf
        new_decomp = OrderedDict{PEPFunction,Float64}(pf => Float64(s))
        return PEPFunction(is_leaf=false, decomposition_dict=new_decomp, reuse_gradient=pf.reuse_gradient)
    else
        new_decomp = OrderedDict{PEPFunction,Float64}(key => value * s for (key, value) in pf.decomposition_dict)
        return PEPFunction(is_leaf=false, decomposition_dict=new_decomp, reuse_gradient=pf.reuse_gradient)
    end
end

-(f::AbstractFunction) = Base.:*(-1, f)
-(f1::AbstractFunction, f2::AbstractFunction) = f1 + (-(f2))
*(f::AbstractFunction, s::Real) = s * f
/(f::AbstractFunction, s::Real) = f * (1 / s)


function add_class_constraints!(func::PEPFunction)
    error("This method must be overwritten in by a concrete PEPFunction subtype (NotImplementedError)")
end


add_constraint!(func::PEPFunction, constraint::Constraint) =
    push!(func.list_of_constraints, constraint)


function add_psd_matrix!(func::PEPFunction, matrix_of_expressions)
    push!(func.list_of_psd, PSDMatrix(matrix_of_expressions))
    return func.list_of_psd[end]
end

add_psd_matrix!(f::AbstractFunction, matrix_of_expressions) =
    add_psd_matrix!(_get_pep_func(f), matrix_of_expressions)


_is_already_evaluated_on_point(func::PEPFunction, point::Point) = begin
    for triplet in func.list_of_points
        if triplet[1].decomposition_dict == point.decomposition_dict
            return (triplet[2], triplet[3])
        end
    end
    nothing
end


function _separate_leaf_functions_regarding_their_need_on_point(func::PEPFunction, point::Point)
    list_nothing, list_grad_only, list_grad_val = [], [], []
    for (f, weight) in func.decomposition_dict
        if _is_already_evaluated_on_point(f, point) !== nothing
            if f.reuse_gradient
                push!(list_nothing, (f, weight))
            else
                push!(list_grad_only, (f, weight))
            end
        else
            push!(list_grad_val, (f, weight))
        end
    end
    return list_nothing, list_grad_only, list_grad_val
end


function add_point!(func::PEPFunction, triplet::Tuple{Point,Point,Expression})
    point, g, f = triplet
    point.decomposition_dict, g.decomposition_dict, f.decomposition_dict =
        map(prune_dict, (point.decomposition_dict, g.decomposition_dict, f.decomposition_dict))

    push!(func.list_of_points, triplet)

    isempty(g.decomposition_dict) &&
        push!(func.list_of_stationary_points, triplet)

    if !func._is_leaf

        func.decomposition_dict = prune_dict(func.decomposition_dict)
        list_nothing, list_grad_only, list_grad_val =
            _separate_leaf_functions_regarding_their_need_on_point(func, point)
        list_needs_something = vcat(list_grad_only, list_grad_val)

        if !isempty(list_needs_something)
            all_leafs = vcat(list_nothing, list_needs_something)
            grad_last, val_last = g, f

            for i in 1:(length(all_leafs)-1)
                current_func, current_weight = all_leafs[i]
                grad, val = oracle!(current_func, point)
                grad_last -= current_weight * grad
                val_last  -= current_weight * val
            end

            last_func, last_weight = all_leafs[end]
            add_point!(last_func, (point,
                                   grad_last / last_weight,
                                   val_last  / last_weight))
        end
    end
end


function oracle!(func::PEPFunction, point::Point)
    evaluation = _is_already_evaluated_on_point(func, point)
    if evaluation !== nothing && func.reuse_gradient
        return evaluation
    end

    f = (evaluation !== nothing && !func.reuse_gradient) ? evaluation[2] : nothing
    list_nothing, list_grad_only, list_grad_val = _separate_leaf_functions_regarding_their_need_on_point(func, point)

    if f === nothing
        f = if isempty(list_grad_val)
            f_agg = Expression(is_leaf=false, decomposition_dict=OrderedDict{Any,Float64}())
            for (f_leaf, w) in func.decomposition_dict
                f_agg += w * value!(f_leaf, point)
            end
            f_agg
        else
            Expression()
        end
    end

    g = if isempty(list_grad_val) && isempty(list_grad_only)
        g_agg = Point(is_leaf=false, decomposition_dict=OrderedDict{Point,Float64}())
        for (f_leaf, w) in func.decomposition_dict
            g_agg += w * gradient!(f_leaf, point)
        end
        g_agg
    else
        Point()
    end

    add_point!(func, (point, g, f))
    return g, f
end


value!(func::PEPFunction, point::Point) = (
    _is_already_evaluated_on_point(func, point) !== nothing ?
        _is_already_evaluated_on_point(func, point)[2]
      : oracle!(func, point)[2]
)

gradient!(func::PEPFunction, point::Point) = oracle!(func, point)[1]

subgradient!(func::PEPFunction, point::Point) = gradient!(func, point)


function stationary_point!(func::PEPFunction; return_gradient_and_function_value=false)

    point, g, f =
    Point(),
    Point(is_leaf=false,
        decomposition_dict=OrderedDict{Point,Float64}()),
    Expression()

    add_point!(func, (point, g, f))

    return return_gradient_and_function_value ? (point, g, f) : point

end


function fixed_point!(func::PEPFunction)
    x = Point()
    fx = Expression()
    add_point!(func, (x, x, fx))
    return x, x, fx
end


oracle!(f::AbstractFunction, p::Point) = oracle!(_get_pep_func(f), p)

gradient!(f::AbstractFunction, p::Point) = gradient!(_get_pep_func(f), p)

value!(f::AbstractFunction, p::Point) = value!(_get_pep_func(f), p)

stationary_point!(f::AbstractFunction) = stationary_point!(_get_pep_func(f))
