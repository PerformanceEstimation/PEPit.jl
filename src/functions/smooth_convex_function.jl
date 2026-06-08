@doc raw"""
    SmoothConvexFunction(param; reuse_gradient=true)

Represent the `SmoothConvexFunction` interpolation class in PEPit.jl.

Implement interpolation constraints of the class of smooth convex functions.

# Class parameters
- `L`: smoothness parameter

Smooth convex functions are characterized by the smoothness parameter `L`, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, SmoothConvexFunction, param)
```

# Fields
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct SmoothConvexFunction <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function SmoothConvexFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(param["L"], func)
    end
end


gradient!(f::SmoothConvexFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothConvexFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothConvexFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothConvexFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothConvexFunction)
    points_list = func._PEPit_func.list_of_points
    for (i, point_i) in enumerate(points_list), (j, point_j) in enumerate(points_list)
        if i == j
            continue
        end
        xi, gi, fi = point_i
        xj, gj, fj = point_j
        constraint = (fi - fj >= gj * (xi - xj) + 1 / (2 * func.L) * (gi - gj)^2)
        add_constraint!(func, constraint)
    end
end


_get_pep_func(f::SmoothConvexFunction) = f._PEPit_func
