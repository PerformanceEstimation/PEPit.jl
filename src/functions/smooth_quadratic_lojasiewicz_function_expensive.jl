@doc raw"""
    SmoothQuadraticLojasiewiczFunctionExpensive(param; reuse_gradient=true)

Class of ``L``-smooth (not necessarily convex) functions that also satisfy a
quadratic Łojasiewicz inequality (sometimes also referred to as a
Polyak-Łojasiewicz inequality) with parameter ``\mu``. Extensive descriptions
of such classes of functions can be found in [1, 2]. The conditions
implemented here are presented in [3, Proposition 3.4]; compared with
[`SmoothQuadraticLojasiewiczFunctionCheap`](@ref), they are tighter but more
expensive (two ``2 \times 2`` PSD blocks per ordered pair of oracle points).

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

# Necessary conditions
A stationary point ``(x_\star, g_\star = 0, f_\star)`` is created automatically
if none was requested through [`stationary_point!`](@ref). Associating with
each oracle call ``i`` the triplet ``(x_i, g_i, f_i)``, the two-sided
Łojasiewicz bounds are added first:

```math
\frac{1}{2L} \|g_i\|^2 \leqslant f_i - f_\star \leqslant \frac{1}{2\mu} \|g_i\|^2
\qquad \text{for all } i \text{ with } x_i \neq x_\star.
```

Then, for every ordered pair ``i \neq j``, defining the negated smoothness
surplus and the Łojasiewicz surpluses

```math
\begin{aligned}
A & = -(f_i - f_j) + \frac{1}{2} \langle g_i + g_j, x_i - x_j \rangle
+ \frac{1}{4L} \|g_i - g_j\|^2 - \frac{L}{4} \|x_i - x_j\|^2, \\
B & = (L + \mu) \left( f_i - f_\star - \frac{1}{2L} \|g_i\|^2 \right), \qquad
C = (L - \mu) \left( f_j - f_\star + \frac{1}{2L} \|g_j\|^2 \right),
\end{aligned}
```

two slack [`Expression`](@ref)s ``s_{12}, s_{22}`` are created and the two
coupled PSD constraints of [3, Proposition 3.4] are imposed, with
``D = B - C - (L + 3\mu) A``:

```math
\begin{pmatrix} -(2L+\mu) A & s_{12} \\ s_{12} & s_{22} \end{pmatrix} \succeq 0,
\quad
\begin{pmatrix}
-(2L+\mu) A - \frac{4\mu}{2L+\mu} s_{12} - D &
s_{12} - \frac{\mu}{2L+\mu} s_{22} - \frac{L+\mu}{2} A + B \\
s_{12} - \frac{\mu}{2L+\mu} s_{22} - \frac{L+\mu}{2} A + B &
s_{22} - B
\end{pmatrix} \succeq 0.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1, "L" => 1.0)
f = declare_function!(problem, SmoothQuadraticLojasiewiczFunctionExpensive, param)
```

!!! note
    Smooth functions are necessarily differentiable, hence `reuse_gradient` is
    set to `true`.

# Fields
- `mu::Float64`: quadratic Łojasiewicz parameter ``\mu``.
- `L::Float64`: smoothness parameter ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] S. Lojasiewicz (1963).
Une propriété topologique des sous-ensembles analytiques réels.
Les équations aux dérivées partielles, 117 (1963), 87-89.](https://aif.centre-mersenne.org/item/10.5802/aif.1384.pdf)

[[2] J. Bolte, A. Daniilidis, and A. Lewis (2007).
The Łojasiewicz inequality for nonsmooth subanalytic functions with
applications to subgradient dynamical systems. SIAM Journal on Optimization 17,
1205-1223.](https://bolte.perso.math.cnrs.fr/Loja.pdf)

[[3] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
A constructive approach to strengthen algebraic descriptions of function and
operator classes.](https://arxiv.org/pdf/2504.14377.pdf)

See also [`declare_function!`](@ref), [`stationary_point!`](@ref),
[`SmoothFunction`](@ref), and [`SmoothQuadraticLojasiewiczFunctionCheap`](@ref).
"""
mutable struct SmoothQuadraticLojasiewiczFunctionExpensive <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function SmoothQuadraticLojasiewiczFunctionExpensive(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        @assert 0 <= mu <= L
        return new(mu, L, func)
    end
end

gradient!(f::SmoothQuadraticLojasiewiczFunctionExpensive, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothQuadraticLojasiewiczFunctionExpensive, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothQuadraticLojasiewiczFunctionExpensive) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothQuadraticLojasiewiczFunctionExpensive, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothQuadraticLojasiewiczFunctionExpensive)
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


    _, _, fs = stationary[1]
    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        A = -fi + fj + 1 / 2 * (gi + gj) * (xi - xj) + 1 / (4 * L) * (gi - gj)^2 - L / 4 * (xi - xj)^2
        B = (L + mu) * (fi - fs - 1 / (2 * L) * gi^2)
        C = (L - mu) * (fj - fs + 1 / (2 * L) * gj^2)

        Mt11 = -A * (2 * L + mu)
        Mt12 = Expression()
        Mt22 = Expression()

        D = B - C - (L + 3 * mu) * A
        M11 = Mt11 - 4 * mu / (2 * L + mu) * Mt12 - D
        M12 = Mt12 - mu / (2 * L + mu) * Mt22 - (L + mu) / 2 * A + B
        M22 = Mt22 - B

        T1 = Matrix{Expression}(undef, 2, 2)
        T1[1, 1] = M11; T1[1, 2] = M12; T1[2, 1] = M12; T1[2, 2] = M22
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T1))

        T2 = Matrix{Expression}(undef, 2, 2)
        T2[1, 1] = Mt11; T2[1, 2] = Mt12; T2[2, 1] = Mt12; T2[2, 2] = Mt22
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T2))
    end
end

_get_pep_func(f::SmoothQuadraticLojasiewiczFunctionExpensive) = f._PEPit_func
