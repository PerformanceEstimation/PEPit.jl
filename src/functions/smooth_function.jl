@doc raw"""
    SmoothFunction(param; reuse_gradient=true)

Represent the `SmoothFunction` interpolation class in PEPit.jl.

Implement the interpolation constraints of the class of smooth (not necessarily convex) functions.

# Class parameters
- `L`: smoothness parameter

Smooth functions are characterized by the smoothness parameter `L`, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, SmoothFunction, param)
```

# Fields
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct SmoothFunction <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function SmoothFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(param["L"], func)
    end
end


gradient!(f::SmoothFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothFunction)
    points_list = func._PEPit_func.list_of_points
    for (i, point_i) in enumerate(points_list), (j, point_j) in enumerate(points_list)
        if i == j
            continue
        end
        xi, gi, fi = point_i
        xj, gj, fj = point_j
        constraint = (
            fi - fj >=
            -func.L / 4 * (xi - xj)^2 +
            1 / 2 * (gi + gj) * (xi - xj) +
            1 / (4 * func.L) * (gi - gj)^2
        )
        add_constraint!(func, constraint)
    end
end


_get_pep_func(f::SmoothFunction) = f._PEPit_func
