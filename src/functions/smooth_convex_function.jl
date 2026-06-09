@doc raw"""
    SmoothConvexFunction(param; reuse_gradient=true)

Interpolation class of ``L``-smooth convex functions.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["L"]`: smoothness parameter ``L``.

# Interpolation conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)`` of
point, gradient, and function value, the following constraint is added for
every pair ``i \neq j`` (see [1, Theorem 4] with ``\mu = 0``):

```math
f_i - f_j \geqslant \langle g_j, x_i - x_j \rangle + \frac{1}{2L} \|g_i - g_j\|^2.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)
f = declare_function!(problem, SmoothConvexFunction, param)
```

!!! note
    Smooth convex functions are necessarily differentiable, hence
    `reuse_gradient` is set to `true`. To model a convex function that is not
    smooth, use [`ConvexFunction`](@ref) rather than `L == Inf`.

# Fields
- `L::Float64`: smoothness parameter ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Smooth strongly convex interpolation and exact worst-case performance of
first-order methods. Mathematical Programming, 161(1-2), 307-345.](https://arxiv.org/pdf/1502.05666.pdf)

See also [`declare_function!`](@ref), [`ConvexFunction`](@ref), and
[`SmoothStronglyConvexFunction`](@ref).
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
