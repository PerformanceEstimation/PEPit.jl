"""
    Point(; is_leaf=true, decomposition_dict=nothing)

Represent an element of the ambient Hilbert space used by a performance
estimation problem.

`Point` objects encode both iterates and gradients/subgradients. A leaf point
is an independent symbolic vector and receives its own row/column in the Gram
matrix. A non-leaf point stores a linear combination of leaf points through
`decomposition_dict`; no new Gram coordinate is introduced.

# Fields
- `_id`: unique object identifier used for hashing and equality.
- `_is_leaf`: whether the point is an independent symbolic vector.
- `decomposition_dict`: coefficients of the point in the leaf-point basis.
- `counter`: Gram-matrix index for leaf points, or `nothing` for non-leaves.
- `_value`: numerical vector recovered from the solved Gram matrix.

# Mathematical model
If `p` and `q` are `Point`s, then `p + q`, `p - q`, and scalar multiples are
again symbolic points. The product `p * q` creates an [`Expression`](@ref)
representing the inner product `\\langle p, q \\rangle`, and `p^2` represents
`\\|p\\|^2`.

# Examples
```julia
x0 = Point()
x1 = Point()
direction = (x1 - x0) / 2
squared_norm = direction^2
```

See also [`Expression`](@ref), [`PEP`](@ref), [`set_initial_point!`](@ref), and
[`evaluate`](@ref).
"""
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


"""
    get_is_leaf(obj)

Return whether `obj` is a leaf symbolic object with its own SDP variable index.

For points, leaf objects are the independent vectors that generate the Gram
matrix. For expressions, leaf objects are independent scalar function values.
Non-leaf objects are affine or linear combinations of existing leaves.
"""
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


"""
    evaluate(obj)

Return the numerical value assigned to a symbolic object after the PEP has been
solved. For a [`Point`](@ref), the value is recovered from a Gram matrix
factorization.

For a non-leaf point, the value is reconstructed from its leaf decomposition.
Calling this before [`solve!`](@ref) or [`solve_dual!`](@ref) has populated the
numerical realization raises an error.
"""
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
