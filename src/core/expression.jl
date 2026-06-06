mutable struct Expression <: AbstractExpression
    _id::Int
    _is_leaf::Bool
    decomposition_dict::OrderedDict{Any,Float64}
    counter::Union{Int,Nothing}
    _value::Union{Float64,Nothing}

    function Expression(is_leaf::Bool, decomposition_dict::Union{OrderedDict{Any,Float64},Nothing})
        if is_leaf
            @assert decomposition_dict === nothing
            expr = new((NEXT_ID[] += 1), true, OrderedDict{Any,Float64}(), Expression_counter[], nothing)
            expr.decomposition_dict[expr] = 1.0
            Expression_counter[] += 1
            push!(GLOBAL_LEAF_EXPRESSIONS, expr)
            return expr
        else
            @assert decomposition_dict isa OrderedDict
            return new((NEXT_ID[] += 1), false, decomposition_dict, nothing, nothing)
        end
    end
end


Expression(; is_leaf=true, decomposition_dict=nothing) = Expression(is_leaf, decomposition_dict)


Base.hash(e::Expression, h::UInt) = hash(e._id, h)

Base.isequal(e1::Expression, e2::Expression) = e1._id == e2._id

Base.:(==)(e1::Expression, e2::Expression) = Constraint(e1 - e2, "equality")


Base.isequal(::Expression, ::Real) = false
Base.isequal(::Real, ::Expression) = false
Base.isequal(::Expression, ::Tuple{Point,Point}) = false
Base.isequal(::Tuple{Point,Point}, ::Expression) = false


get_is_leaf(e::Expression) = e._is_leaf


+(e1::Expression, e2::Expression) = Expression(is_leaf=false, decomposition_dict=merge_dicts(e1.decomposition_dict, e2.decomposition_dict))

+(e::Expression, s::Real) = Expression(is_leaf=false, decomposition_dict=merge_dicts(e.decomposition_dict, OrderedDict{Any,Float64}(1 => Float64(s))))

+(s::Real, e::Expression) = e + s

-(e::Expression) = Expression(is_leaf=false, decomposition_dict=OrderedDict{Any,Float64}(k => -v for (k, v) in e.decomposition_dict))

-(e1::Expression, e2::Expression) = e1 + (-e2)

-(e::Expression, s::Real) = e + (-s)

-(s::Real, e::Expression) = s + (-e)

*(s::Real, e::Expression) = Expression(is_leaf=false, decomposition_dict=OrderedDict{Any,Float64}(k => v * s for (k, v) in e.decomposition_dict))

*(e::Expression, s::Real) = s * e

/(e::Expression, s::Real) = e * (1 / s)


Expression(e::Expression) = e

Expression(s::Real) = Expression(
    is_leaf=false,
    decomposition_dict=OrderedDict{Any,Float64}(1 => Float64(s))
)


Base.:(<)(e1::Expression, e2::Expression) = (@warn "[⚠️] Strict constraints will lead to the same solution as non-strict"; e1 <= e2)

Base.:(<)(e1::Expression, e2::Real) = (@warn "[⚠️] Strict constraints will lead to the same solution as non-strict"; e1 <= e2)

Base.:(<)(e1::Real, e2::Expression) = (@warn "[⚠️] Strict constraints will lead to the same solution as non-strict"; e1 <= e2)

Base.:(>)(e1::Expression, e2::Expression) = (@warn "[⚠️] Strict constraints will lead to the same solution as non-strict"; e1 >= e2)

Base.:(>)(a::Expression, b::Real) = (@warn "[⚠️] Strict constraints lead to same solution"; a >= b)

Base.:(>)(a::Real, b::Expression) = (@warn "[⚠️] Strict constraints lead to same solution"; a >= b)


const null_expression = Expression(
    is_leaf=false,
    decomposition_dict=OrderedDict{Any,Float64}()
)


function evaluate(e::Expression)
    isnothing(e._value) || return e._value
    if get_is_leaf(e)
        error("[💀] The PEP must be solved to evaluate Expressions!")
    end
    val = 0.0
    for (key, weight) in e.decomposition_dict
        if key isa Expression
            @assert get_is_leaf(key) "[💀] Non-leaf Expression used as a key; only leaf function values are allowed."

            val += weight * evaluate(key)

        elseif key isa Tuple{Point,Point}
            p1, p2 = key
            @assert get_is_leaf(p1) && get_is_leaf(p2) "[💀] Inner-product keys must be tuples of leaf Points."
            val += weight * dot(evaluate(p1), evaluate(p2))

        elseif key == 1
            val += weight

        else
            error("[💀] Expressions are made of function values (leaf Expressions), inner products of leaf Points, and constants only. Got $(typeof(key)).")
        end
    end
    e._value = val
    return val
end
