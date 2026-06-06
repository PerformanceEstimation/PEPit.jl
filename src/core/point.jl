mutable struct Point <: AbstractPoint
    _id::Int
    _is_leaf::Bool
    decomposition_dict::OrderedDict{Point,Float64}
    counter::Union{Int,Nothing}
    _value::Union{Vector{Float64},Nothing}

    function Point(is_leaf::Bool, decomposition_dict::Union{OrderedDict{Point,Float64},Nothing})
        if is_leaf
            @assert decomposition_dict === nothing
            p = new((NEXT_ID[] += 1), true, OrderedDict{Point,Float64}(), Point_counter[], nothing)
            p.decomposition_dict[p] = 1.0
            Point_counter[] += 1
            push!(GLOBAL_LEAF_POINTS, p)
            return p
        else
            @assert decomposition_dict isa OrderedDict
            return new((NEXT_ID[] += 1), false, decomposition_dict, nothing, nothing)
        end
    end
end


Point(; is_leaf=true, decomposition_dict=nothing) = Point(is_leaf, decomposition_dict)


Base.hash(p::Point, h::UInt) = hash(p._id, h)

Base.:(==)(p1::Point, p2::Point) = p1._id == p2._id

Base.isequal(p1::Point, p2::Point) = p1._id == p2._id


get_is_leaf(p::Point) = p._is_leaf


+(p1::Point, p2::Point) = Point(is_leaf=false, decomposition_dict=prune_dict(merge_dicts(p1.decomposition_dict, p2.decomposition_dict)))


-(p::Point) = Point(is_leaf=false, decomposition_dict=OrderedDict{Point,Float64}(key => -value for (key, value) in p.decomposition_dict))


-(p1::Point, p2::Point) = p1 + (-p2)


*(s::Real, p::Point) = Point(is_leaf=false, decomposition_dict=OrderedDict{Point,Float64}(key => value * s for (key, value) in p.decomposition_dict))


*(p::Point, s::Real) = s * p


/(p::Point, s::Real) = p * (1 / s)


*(p1::Point, p2::Point) = Expression(is_leaf=false, decomposition_dict=multiply_dicts(p1.decomposition_dict, p2.decomposition_dict))


^(p::Point, power::Int) = (@assert power == 2; p * p)


const null_point = Point(is_leaf=false, decomposition_dict=OrderedDict{Point,Float64}())


function evaluate(p::Point)
    isnothing(p._value) || return p._value
    if get_is_leaf(p)
        error("The PEP must be solved to evaluate Points!")
    end

    p._value = sum(
        weight * evaluate(point) for (point, weight) in p.decomposition_dict;
        init = zeros(Float64, Point_counter[])
    )
    return p._value
end
