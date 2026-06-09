@doc raw"""
    ConvexSupportFunction(param=OrderedDict(); reuse_gradient=false)

Interpolation class of closed convex support functions, optionally with a
bounded Lipschitz constant ``M`` (equivalently, support functions of sets
contained in a ball of radius ``M``).

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["M"]` (optional, default `Inf`): upper bound ``M`` on the Lipschitz constant.

# Interpolation conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)``, where
``g_i`` is a subgradient of the support function at ``x_i`` (that is, a
maximizing element of the underlying set), the following constraints are added
(see [1]):

```math
\begin{aligned}
\langle g_i, x_i \rangle - f_i & = 0 && \text{for all } i, \\
\|g_i\|^2 & \leqslant M^2 && \text{for all } i \text{ (added when } M < \infty\text{)}, \\
\langle x_j, g_i - g_j \rangle & \leqslant 0 && \text{for all } i \neq j.
\end{aligned}
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("M" => 1.0)
f = declare_function!(problem, ConvexSupportFunction, param)
```

# Fields
- `M::Float64`: Lipschitz constant bound ``M``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex
optimization. SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

See also [`declare_function!`](@ref), [`ConvexFunction`](@ref), and
[`ConvexIndicatorFunction`](@ref).
"""
mutable struct ConvexSupportFunction <: AbstractFunction
    M::Float64
    _PEPit_func::PEPFunction

    function ConvexSupportFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        M = haskey(param, "M") ? float(param["M"]) : Inf
        return new(M, func)
    end
end

gradient!(f::ConvexSupportFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexSupportFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexSupportFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexSupportFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexSupportFunction)
    points = func._PEPit_func.list_of_points


    for (xi, gi, fi) in points
        add_constraint!(func, gi * xi - fi == 0)
    end


    if func.M != Inf
        M2 = func.M^2
        for (xi, gi, fi) in points
            add_constraint!(func, gi^2 <= M2)
        end
    end


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, xj * (gi - gj) <= 0)
    end
end

_get_pep_func(f::ConvexSupportFunction) = f._PEPit_func
