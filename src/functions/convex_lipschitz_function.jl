@doc raw"""
    ConvexLipschitzFunction(param; reuse_gradient=false)

Interpolation class of convex, closed, and proper (CCP) functions that are
``M``-Lipschitz continuous.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["M"]`: Lipschitz continuity parameter ``M``.

# Interpolation conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)`` of
point, subgradient, and function value, the following constraints are added
(see [1]):

```math
\begin{aligned}
f_i - f_j & \geqslant \langle g_j, x_i - x_j \rangle && \text{for all } i \neq j, \\
\|g_i\|^2 & \leqslant M^2 && \text{for all } i \text{ (added when } M < \infty\text{)}.
\end{aligned}
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("M" => 1.0)
f = declare_function!(problem, ConvexLipschitzFunction, param)
```

!!! note
    With `M == Inf` the Lipschitz bound adds no constraint, so the class
    coincides with [`ConvexFunction`](@ref); the constructor emits a warning in
    that case.

# Fields
- `M::Float64`: Lipschitz continuity parameter ``M``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex
optimization. SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

See also [`declare_function!`](@ref), [`ConvexFunction`](@ref), and
[`SmoothConvexLipschitzFunction`](@ref).
"""
mutable struct ConvexLipschitzFunction <: AbstractFunction
    M::Float64
    _PEPit_func::PEPFunction

    function ConvexLipschitzFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        M = float(param["M"])
        if M == Inf
            @warn "(PEPit) The class of convex M-Lipschitz functions with M == Inf implies no constraint: it contains all convex closed proper functions."
        end
        return new(M, func)
    end
end


gradient!(f::ConvexLipschitzFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexLipschitzFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexLipschitzFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexLipschitzFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexLipschitzFunction)
    points_list = func._PEPit_func.list_of_points


    if func.M != Inf
        M2 = func.M^2
        for point_i in points_list
            _, gi, _ = point_i
            add_constraint!(func, gi^2 <= M2)
        end
    end


    for point_i in points_list, point_j in points_list
        if point_i == point_j
            continue
        end
        xi, gi, fi = point_i
        xj, gj, fj = point_j
        add_constraint!(func, fi - fj >= gj * (xi - xj))
    end
end

_get_pep_func(f::ConvexLipschitzFunction) = f._PEPit_func
