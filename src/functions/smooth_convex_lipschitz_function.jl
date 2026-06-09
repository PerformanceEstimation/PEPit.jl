@doc raw"""
    SmoothConvexLipschitzFunction(param; reuse_gradient=true)

Interpolation class of ``L``-smooth convex functions that are ``M``-Lipschitz
continuous.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["L"]`: smoothness parameter ``L``.
- `param["M"]`: Lipschitz continuity parameter ``M``.

# Interpolation conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)`` of
point, gradient, and function value, the following constraints are added
(see [1]):

```math
\begin{aligned}
f_i - f_j & \geqslant \langle g_j, x_i - x_j \rangle + \frac{1}{2L} \|g_i - g_j\|^2
&& \text{for all } i \neq j, \\
\|g_i\|^2 & \leqslant M^2 && \text{for all } i.
\end{aligned}
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0, "M" => 1.0)
f = declare_function!(problem, SmoothConvexLipschitzFunction, param)
```

!!! note
    Smooth convex Lipschitz continuous functions are necessarily
    differentiable, hence `reuse_gradient` is set to `true`. To drop the
    smoothness requirement, use [`ConvexLipschitzFunction`](@ref) rather than
    `L == Inf`.

# Fields
- `L::Float64`: smoothness parameter ``L``.
- `M::Float64`: Lipschitz continuity parameter ``M``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex
optimization. SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

See also [`declare_function!`](@ref), [`SmoothConvexFunction`](@ref), and
[`ConvexLipschitzFunction`](@ref).
"""
mutable struct SmoothConvexLipschitzFunction <: AbstractFunction
    L::Float64
    M::Float64
    _PEPit_func::PEPFunction

    function SmoothConvexLipschitzFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["L"]), float(param["M"]), func)
    end
end

gradient!(f::SmoothConvexLipschitzFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothConvexLipschitzFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothConvexLipschitzFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothConvexLipschitzFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothConvexLipschitzFunction)
    points = func._PEPit_func.list_of_points


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj) + 1 / (2 * func.L) * (gi - gj)^2)
    end


    M2 = func.M^2
    for (xi, gi, fi) in points
        add_constraint!(func, gi^2 <= M2)
    end
end

_get_pep_func(f::SmoothConvexLipschitzFunction) = f._PEPit_func
