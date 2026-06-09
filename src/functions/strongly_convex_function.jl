@doc raw"""
    StronglyConvexFunction(param; reuse_gradient=false)

Interpolation class of ``\mu``-strongly convex closed proper functions
(strongly convex functions whose epigraphs are non-empty closed sets).

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["mu"]`: strong convexity parameter ``\mu``.

# Interpolation conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)`` of
point, subgradient, and function value, the following constraint is added for
every pair ``i \neq j`` (see [1]):

```math
f_i - f_j \geqslant \langle g_j, x_i - x_j \rangle + \frac{\mu}{2} \|x_i - x_j\|^2.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1)
f = declare_function!(problem, StronglyConvexFunction, param)
```

# Fields
- `mu::Float64`: strong convexity parameter ``\mu``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Smooth strongly convex interpolation and exact worst-case performance of
first-order methods. Mathematical Programming, 161(1-2), 307-345.](https://arxiv.org/pdf/1502.05666.pdf)

See also [`declare_function!`](@ref), [`ConvexFunction`](@ref), and
[`SmoothStronglyConvexFunction`](@ref).
"""
mutable struct StronglyConvexFunction <: AbstractFunction
    mu::Float64
    _PEPit_func::PEPFunction

    function StronglyConvexFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(param["mu"], func)
    end
end

gradient!(f::StronglyConvexFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::StronglyConvexFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::StronglyConvexFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::StronglyConvexFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::StronglyConvexFunction)
    points_list = func._PEPit_func.list_of_points
    for point_i in points_list, point_j in points_list
        if point_i == point_j
            continue
        end
        xi, gi, fi = point_i
        xj, gj, fj = point_j
        constraint = (fi - fj >= gj * (xi - xj) + func.mu / 2 * (xi - xj)^2)
        add_constraint!(func, constraint)
    end
end

_get_pep_func(f::StronglyConvexFunction) = f._PEPit_func
