@doc raw"""
    ConvexFunction(param; <keyword arguments>)

Represent the `ConvexFunction` interpolation class in PEPit.jl.

Implement the interpolation constraints of the class of convex, closed and proper (CCP) functions (i.e., convex
functions whose epigraphs are non-empty closed sets).

General CCP functions are not characterized by any parameter, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, ConvexFunction, param)
```

# Fields
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
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
