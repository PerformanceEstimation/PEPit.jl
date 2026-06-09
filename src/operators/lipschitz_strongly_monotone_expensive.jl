@doc raw"""
    LipschitzStronglyMonotoneOperatorExpensive(param; reuse_gradient=true)

Class of ``L``-Lipschitz continuous and ``\mu``-strongly monotone (maximally
monotone) operators, modeled through the strengthened necessary constraints of
[1, Proposition 3.15] (details in [1, Appendix E]), which are stronger than
those used in [2] (and in [`LipschitzStronglyMonotoneOperatorCheap`](@ref))
but significantly more expensive (two ``7 \times 7`` PSD blocks per ordered
triplet of oracle points).

Overrides `add_class_constraints!` to add the conditions of the class when
[`solve!`](@ref) builds the SDP.

!!! warning
    Lipschitz strongly monotone operators do not enjoy known interpolation
    conditions. The conditions implemented in this class are necessary but a
    priori not sufficient for interpolation. Hence, the numerical results
    obtained when using this class might be non-tight upper bounds (see
    Discussions in [1, Section 2]).

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["mu"]`: strong monotonicity parameter ``\mu``.
- `param["L"]`: Lipschitz continuity parameter ``L``.

# Necessary conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where ``g_i``
denotes the operator value at ``x_i``, the implementation considers, for every
ordered triplet of oracle points ``(i, j, k)`` (not all equal), the pairwise
Lipschitz residuals and scaled strong-monotonicity residuals

```math
A_{pq} = \|g_p - g_q\|^2 - L^2 \|x_p - x_q\|^2, \qquad
B_{pq} = 2L \left( -\langle g_p - g_q, x_p - x_q \rangle + \mu \|x_p - x_q\|^2 \right),
```

over the pairs ``(p, q) \in \{(i,j), (i,k), (j,k)\}``. Two ``7 \times 7``
matrices are built from these residuals together with nine free slack
[`Expression`](@ref)s (the two matrices differ by swapping the roles of the
two residual families), and both are constrained to be PSD through
[`PSDMatrix`](@ref) objects, following [1, Proposition 3.15]. See the
implementation of `add_class_constraints!` in this file for the exact entries.

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1, "L" => 1.0)
op = declare_function!(problem, LipschitzStronglyMonotoneOperatorExpensive, param)
```

!!! note
    With `L == Inf` the Lipschitz bound adds no constraint, so the class
    reduces to [`StronglyMonotoneOperator`](@ref); the constructor emits a
    warning in that case.

# Fields
- `mu::Float64`: strong monotonicity parameter ``\mu``.
- `L::Float64`: Lipschitz continuity parameter ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
A constructive approach to strengthen algebraic descriptions of function and
operator classes.](https://arxiv.org/pdf/2504.14377.pdf)

[[2] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

See also [`declare_function!`](@ref), [`LipschitzOperator`](@ref),
[`StronglyMonotoneOperator`](@ref), and
[`LipschitzStronglyMonotoneOperatorCheap`](@ref).
"""
mutable struct LipschitzStronglyMonotoneOperatorExpensive <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function LipschitzStronglyMonotoneOperatorExpensive(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        L == Inf && @warn "(PEPit) L == Inf; consider StronglyMonotoneOperator instead."
        return new(mu, L, func)
    end
end

add_constraint!(op::LipschitzStronglyMonotoneOperatorExpensive, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::LipschitzStronglyMonotoneOperatorExpensive) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::LipschitzStronglyMonotoneOperatorExpensive)
    internal = op._PEPit_func
    pts = internal.list_of_points
    L = op.L; mu = op.mu

    build_T = function (xi, ti, xj, tj, xk, tk, opt)
        if opt == 1
            Aij = (ti - tj)^2 - L^2 * (xi - xj)^2
            Aik = (ti - tk)^2 - L^2 * (xi - xk)^2
            Ajk = (tk - tj)^2 - L^2 * (xk - xj)^2
            Bij = 2 * L * (-(ti - tj) * (xi - xj) + mu * (xi - xj)^2)
            Bik = 2 * L * (-(ti - tk) * (xi - xk) + mu * (xi - xk)^2)
            Bjk = 2 * L * (-(tk - tj) * (xk - xj) + mu * (xk - xj)^2)
        else
            Bij = (ti - tj)^2 - L^2 * (xi - xj)^2
            Bik = (ti - tk)^2 - L^2 * (xi - xk)^2
            Bjk = (tk - tj)^2 - L^2 * (xk - xj)^2
            Aij = 2 * L * (-(ti - tj) * (xi - xj) + mu * (xi - xj)^2)
            Aik = 2 * L * (-(ti - tk) * (xi - xk) + mu * (xi - xk)^2)
            Ajk = 2 * L * (-(tk - tj) * (xk - xj) + mu * (xk - xj)^2)
        end
        M14 = Expression(); M15 = Expression(); M16 = Expression(); M17 = Expression()
        M26 = Expression(); M27 = Expression(); M34 = Expression(); M37 = Expression(); M46 = Expression()
        M25 = -M14; M23 = -M15; M35 = -M16; M45 = -M27; M56 = -M37; M57 = -M46
        M55 = Aij + 2 * mu * Bij - Ajk - Aik - 2 * M17 - 2 * M26 - 2 * M34
        z = Expression(0.0)
        T = Matrix{Expression}(undef, 7, 7)
        T[1, :] = [-Bij, z, z, M14, M15, M16, M17]
        T[2, :] = [z, -Ajk, M23, z, M25, M26, M27]
        T[3, :] = [z, M23, -Bij, M34, M35, z, M37]
        T[4, :] = [M14, z, M34, -Bjk, M45, M46, z]
        T[5, :] = [M15, M25, M35, M45, M55, M56, M57]
        T[6, :] = [M16, M26, z, M46, M56, -Aik, z]
        T[7, :] = [M17, M27, M37, z, M57, z, -Bik]
        return T
    end

    n = length(pts)
    for i in 1:n, j in 1:n, k in 1:n
        (i == j && i == k) && continue
        xi, ti, _ = pts[i]; xj, tj, _ = pts[j]; xk, tk, _ = pts[k]
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=build_T(xi, ti, xj, tj, xk, tk, 1)))
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=build_T(xi, ti, xj, tj, xk, tk, 0)))
    end
end

_get_pep_func(op::LipschitzStronglyMonotoneOperatorExpensive) = op._PEPit_func
