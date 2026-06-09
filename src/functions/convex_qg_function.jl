@doc raw"""
    ConvexQGFunction(param; reuse_gradient=false)

Interpolation class of convex and quadratically upper bounded
(``\text{QG}^+`` [1]) functions, that is, convex functions satisfying
``f(x) - f_\star \leqslant \frac{L}{2} \|x - x_\star\|^2`` for all ``x``.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["L"]`: quadratic upper bound parameter ``L``.

# Interpolation conditions
A stationary point ``(x_\star, g_\star = 0, f_\star)`` is created automatically
if none was requested through [`stationary_point!`](@ref). Associating with
each oracle call ``i`` the triplet ``(x_i, g_i, f_i)`` of point, subgradient,
and function value, the following constraints are added (see [1]):

```math
\begin{aligned}
f_\star - f_j & \geqslant \langle g_j, x_\star - x_j \rangle + \frac{1}{2L} \|g_j\|^2
&& \text{for all } j \text{ with } x_j \neq x_\star, \\
f_i - f_j & \geqslant \langle g_j, x_i - x_j \rangle && \text{for all } i \neq j.
\end{aligned}
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)
f = declare_function!(problem, ConvexQGFunction, param)
```

# Fields
- `L::Float64`: quadratic upper bound parameter ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] B. Goujaud, A. Taylor, A. Dieuleveut (2022).
Optimal first-order methods for convex functions with a quadratic upper
bound.](https://arxiv.org/pdf/2205.15033.pdf)

See also [`declare_function!`](@ref), [`stationary_point!`](@ref), and
[`ConvexFunction`](@ref).
"""
mutable struct ConvexQGFunction <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function ConvexQGFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["L"]), func)
    end
end

gradient!(f::ConvexQGFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexQGFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexQGFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexQGFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexQGFunction)
    internal = func._PEPit_func
    if isempty(internal.list_of_stationary_points)
        stationary_point!(internal)
    end
    points = internal.list_of_points
    stationary = internal.list_of_stationary_points


    for (xi, gi, fi) in stationary, (xj, gj, fj) in points
        xi == xj && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj) + 1 / (2 * func.L) * gj^2)
    end


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj))
    end
end

_get_pep_func(f::ConvexQGFunction) = f._PEPit_func
