using OrderedCollections

@doc raw"""
    ConvexIndicatorFunction(param=OrderedDict(); reuse_gradient=false)

Interpolation class of closed convex indicator functions, optionally with a
bounded feasible set (through its diameter ``D`` and/or its radius ``R``).

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["D"]` (optional, default `Inf`): upper bound ``D`` on the diameter of the feasible set.
- `param["R"]` (optional, default `Inf`): upper bound ``R`` on the radius of the feasible set.
- `param["center"]` (optional, default `nothing`): center [`Point`](@ref) used by the radius constraint.

# Interpolation conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)``, where
``g_i`` is a subgradient of the indicator at the (feasible) point ``x_i``, the
following constraints are added (see [1]):

```math
\begin{aligned}
f_i & = 0 && \text{for all } i, \\
0 & \geqslant \langle g_j, x_i - x_j \rangle && \text{for all } i \neq j, \\
\|x_i - x_j\|^2 & \leqslant D^2 && \text{for all } i \neq j \text{ (added when } D < \infty\text{)}, \\
\|x_i - c\|^2 & \leqslant R^2 && \text{for all } i \text{ (added when } R < \infty\text{)},
\end{aligned}
```

where ``c`` is the center of the radius constraint.

# Julia usage
```julia
problem = PEP()
param = OrderedDict("D" => 1.0)
f = declare_function!(problem, ConvexIndicatorFunction, param)
```

!!! note
    When `R < Inf` and no `"center"` is supplied, the constructor automatically
    creates a fresh [`Point`](@ref) to act as the (otherwise unspecified)
    center ``c``.

# Fields
- `D::Float64`: diameter bound ``D``.
- `R::Float64`: radius bound ``R``.
- `center::Union{Point,Nothing}`: center of the radius constraint.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex
optimization. SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

See also [`declare_function!`](@ref), [`ConvexFunction`](@ref), and
[`ConvexSupportFunction`](@ref).
"""
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
