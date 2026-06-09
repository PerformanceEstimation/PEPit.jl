@doc raw"""
    SmoothQuadraticLojasiewiczFunctionCheap(param; reuse_gradient=true)

Class of ``L``-smooth (not necessarily convex) functions that also satisfy a
quadratic Łojasiewicz inequality (sometimes also referred to as a
Polyak-Łojasiewicz inequality) with parameter ``\mu``. Extensive descriptions
of such classes of functions can be found in [1, 2]. The conditions
implemented here are presented in [4, Proposition 3.2] (for `alpha` to be
chosen) and [4, Proposition 3.4] with smoothness conditions from [3].

Overrides `add_class_constraints!` to add the conditions of the class when
[`solve!`](@ref) builds the SDP.

!!! warning
    Smooth functions satisfying a Łojasiewicz property do not enjoy known
    interpolation conditions. The conditions implemented in this class are
    necessary but a priori not sufficient for interpolation. Hence, the
    numerical results obtained when using this class might be non-tight upper
    bounds.

# Class parameters
- `param["mu"]`: quadratic Łojasiewicz parameter ``\mu`` (with ``0 \leqslant \mu \leqslant L``).
- `param["L"]`: smoothness parameter ``L``.
- `param["alpha"]` (optional, default `nothing`): relaxation parameter ``\alpha \in [0, 2\mu/(2L+\mu)]``.

# Necessary conditions
A stationary point ``(x_\star, g_\star = 0, f_\star)`` is created automatically
if none was requested through [`stationary_point!`](@ref). Associating with
each oracle call ``i`` the triplet ``(x_i, g_i, f_i)``, the following
constraints are added:

```math
\begin{aligned}
\frac{1}{2L} \|g_i\|^2 \leqslant f_i - f_\star & \leqslant \frac{1}{2\mu} \|g_i\|^2
&& \text{for all } i \text{ with } x_i \neq x_\star, \\
f_i - f_j & \geqslant \frac{1}{2} \langle g_i + g_j, x_i - x_j \rangle
+ \frac{1}{4L} \|g_i - g_j\|^2 - \frac{L}{4} \|x_i - x_j\|^2
&& \text{for all } i \neq j.
\end{aligned}
```

When `alpha` is provided, the smoothness conditions are strengthened, for all
``i \neq j``, into (see [4, Proposition 3.2]):

```math
f_i - f_j \geqslant \frac{1}{2} \langle g_i + g_j, x_i - x_j \rangle
+ \frac{1}{4L} \|g_i - g_j\|^2 - \frac{L}{4} \|x_i - x_j\|^2
+ c_\alpha \left[ (1-\alpha)^2 (L+\mu) \left( f_i - f_\star - \frac{\|g_i\|^2}{2L} \right)
- (L-\mu) \left( f_j - f_\star + \frac{\|g_j\|^2}{2L} \right) \right],
```

with ``c_\alpha = \dfrac{\alpha}{(1-\alpha)\,(2\mu - (L+\mu)\alpha)}``.

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.5, "L" => 1.0, "alpha" => 0.4)
f = declare_function!(problem, SmoothQuadraticLojasiewiczFunctionCheap, param)
```

!!! note
    Smooth functions are necessarily differentiable, hence `reuse_gradient` is
    set to `true`.

# Fields
- `mu::Float64`: quadratic Łojasiewicz parameter ``\mu``.
- `L::Float64`: smoothness parameter ``L``.
- `alpha::Union{Float64,Nothing}`: relaxation parameter ``\alpha``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] S. Lojasiewicz (1963).
Une propriété topologique des sous-ensembles analytiques réels.
Les équations aux dérivées partielles, 117 (1963), 87-89.](https://aif.centre-mersenne.org/item/10.5802/aif.1384.pdf)

[[2] J. Bolte, A. Daniilidis, and A. Lewis (2007).
The Łojasiewicz inequality for nonsmooth subanalytic functions with
applications to subgradient dynamical systems. SIAM Journal on Optimization 17,
1205-1223.](https://bolte.perso.math.cnrs.fr/Loja.pdf)

[[3] A. Taylor, J. Hendrickx, F. Glineur (2017).
Exact worst-case performance of first-order methods for composite convex
optimization. SIAM Journal on Optimization, 27(3):1283-1313.](https://arxiv.org/pdf/1512.07516.pdf)

[[4] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
A constructive approach to strengthen algebraic descriptions of function and
operator classes.](https://arxiv.org/pdf/2504.14377.pdf)

See also [`declare_function!`](@ref), [`stationary_point!`](@ref),
[`SmoothFunction`](@ref), and [`SmoothQuadraticLojasiewiczFunctionExpensive`](@ref).
"""
mutable struct SmoothQuadraticLojasiewiczFunctionCheap <: AbstractFunction
    mu::Float64
    L::Float64
    alpha::Union{Float64,Nothing}
    _PEPit_func::PEPFunction

    function SmoothQuadraticLojasiewiczFunctionCheap(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        alpha = haskey(param, "alpha") && param["alpha"] !== nothing ? float(param["alpha"]) : nothing
        @assert 0 <= mu <= L
        if alpha !== nothing
            @assert 0 <= alpha <= 2 * mu / (2 * L + mu)
        end
        return new(mu, L, alpha, func)
    end
end

gradient!(f::SmoothQuadraticLojasiewiczFunctionCheap, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothQuadraticLojasiewiczFunctionCheap, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothQuadraticLojasiewiczFunctionCheap) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothQuadraticLojasiewiczFunctionCheap, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothQuadraticLojasiewiczFunctionCheap)
    internal = func._PEPit_func
    if isempty(internal.list_of_stationary_points)
        stationary_point!(internal)
    end
    points = internal.list_of_points
    stationary = internal.list_of_stationary_points
    L = func.L; mu = func.mu


    for (xi, gi, fi) in points, (xj, gj, fj) in stationary
        xi == xj && continue
        add_constraint!(func, fi - fj <= 1 / (2 * mu) * gi^2)
    end


    for (xi, gi, fi) in points, (xj, gj, fj) in stationary
        xi == xj && continue
        add_constraint!(func, fi - fj >= 1 / (2 * L) * gi^2)
    end


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, fi - fj >= 1 / 2 * (gi + gj) * (xi - xj) + 1 / (4 * L) * (gi - gj)^2 - L / 4 * (xi - xj)^2)
    end


    if func.alpha !== nothing
        alpha = func.alpha
        _, _, fs = stationary[1]
        const_coef = alpha / (1 - alpha) / (2 * mu - (L + mu) * alpha)
        for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
            i == j && continue
            add_constraint!(func, fi - fj >= 1 / 2 * (gi + gj) * (xi - xj) + 1 / (4 * L) * (gj - gi)^2 - L / 4 * (xj - xi)^2 +
                                  const_coef * ((1 - alpha)^2 * (L + mu) * (fi - fs - gi^2 / 2 / L) - (L - mu) * (fj - fs + gj^2 / 2 / L)))
        end
    end
end

_get_pep_func(f::SmoothQuadraticLojasiewiczFunctionCheap) = f._PEPit_func
