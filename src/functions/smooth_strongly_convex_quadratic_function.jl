@doc raw"""
    SmoothStronglyConvexQuadraticFunction(param; reuse_gradient=true)

Interpolation class of ``L``-smooth ``\mu``-strongly convex quadratic
functions, that is, functions of the form
``f(x) = \frac{1}{2} \langle x - x_\star, H (x - x_\star) \rangle + f_\star``
with ``\mu I \preceq H \preceq L I``.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["mu"]`: strong convexity parameter ``\mu``.
- `param["L"]`: smoothness parameter ``L``.

# Interpolation conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)`` of
point, gradient, and function value, and denoting by
``(x_\star, 0, f_\star)`` the stationary point of the quadratic, the following
constraints are added (see [1]):

```math
\begin{aligned}
f_i - f_\star & = \tfrac{1}{2} \langle g_i, x_i - x_\star \rangle && \text{for all } i, \\
\langle g_j, x_i - x_\star \rangle & = \langle g_i, x_j - x_\star \rangle && \text{for all } i < j,
\end{aligned}
```

together with the PSD constraint ``T \succeq 0``, where ``T`` is the matrix of
expressions with entries

```math
T_{ij} = (L + \mu) \langle g_i, x_j - x_\star \rangle - \langle g_i, g_j \rangle
- \mu L \langle x_i - x_\star, x_j - x_\star \rangle,
```

which is the Gram-space formulation of ``(L I - H)(H - \mu I) \succeq 0``.

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1, "L" => 1.0)
f = declare_function!(problem, SmoothStronglyConvexQuadraticFunction, param)
```

!!! note
    The constructor automatically creates the (unique) stationary point of the
    quadratic; [`stationary_point!`](@ref) returns that same point instead of
    creating a new one. Smooth strongly convex quadratic functions are
    necessarily differentiable, hence `reuse_gradient` is set to `true`.

# Fields
- `mu::Float64`: strong convexity parameter ``\mu``.
- `L::Float64`: smoothness parameter ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] N. Bousselmi, J. Hendrickx, F. Glineur (2023).
Interpolation Conditions for Linear Operators and applications to Performance
Estimation Problems. arXiv preprint.](https://arxiv.org/pdf/2302.08781.pdf)

See also [`declare_function!`](@ref), [`stationary_point!`](@ref),
[`SmoothStronglyConvexFunction`](@ref), and [`PSDMatrix`](@ref).
"""
mutable struct SmoothStronglyConvexQuadraticFunction <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function SmoothStronglyConvexQuadraticFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        stationary_point!(func)
        return new(float(param["mu"]), float(param["L"]), func)
    end
end

gradient!(f::SmoothStronglyConvexQuadraticFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothStronglyConvexQuadraticFunction, p::Point) = value!(f._PEPit_func, p)
add_constraint!(func::SmoothStronglyConvexQuadraticFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)


stationary_point!(f::SmoothStronglyConvexQuadraticFunction) = f._PEPit_func.list_of_stationary_points[1][1]

function add_class_constraints!(func::SmoothStronglyConvexQuadraticFunction)
    internal = func._PEPit_func
    points = internal.list_of_points
    xs, _, fs = internal.list_of_stationary_points[1]


    for (xi, gi, fi) in points
        add_constraint!(func, fi - fs == 0.5 * (xi - xs) * gi)
    end


    n = length(points)
    for i in 1:n, j in (i + 1):n
        xi, gi, _ = points[i]
        xj, gj, _ = points[j]
        add_constraint!(func, (xi - xs) * gj == (xj - xs) * gi)
    end


    if n > 0
        T = Matrix{Expression}(undef, n, n)
        for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
            T[i, j] = (func.L + func.mu) * (gi * (xj - xs)) - gi * gj - (func.mu * func.L) * ((xi - xs) * (xj - xs))
        end
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T))
    end
end

_get_pep_func(f::SmoothStronglyConvexQuadraticFunction) = f._PEPit_func
