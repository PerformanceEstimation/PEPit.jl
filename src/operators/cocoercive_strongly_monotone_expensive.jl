@doc raw"""
    CocoerciveStronglyMonotoneOperatorExpensive(param; reuse_gradient=true)

Class of ``\beta``-cocoercive and ``\mu``-strongly monotone (maximally
monotone) operators, modeled through the strengthened necessary constraints of
[1, Appendix F], which are stronger than those used in [2] (and in
[`CocoerciveStronglyMonotoneOperatorCheap`](@ref)) but significantly more
expensive (two ``7 \times 7`` PSD blocks per ordered triplet of oracle
points).

Overrides `add_class_constraints!` to add the conditions of the class when
[`solve!`](@ref) builds the SDP.

!!! warning
    Those constraints might not be sufficient, thus the characterized class
    might contain more operators.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["mu"]`: strong monotonicity parameter ``\mu``.
- `param["beta"]`: cocoercivity parameter ``\beta``.

# Necessary conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where ``g_i``
denotes the operator value at ``x_i``, the implementation considers, for every
ordered triplet of oracle points ``(i, j, k)`` (not all equal), the pairwise
strong-monotonicity residuals and cocoercivity residuals

```math
A_{pq} = -\langle g_p - g_q, x_p - x_q \rangle + \mu \|x_p - x_q\|^2, \qquad
B_{pq} = -\langle g_p - g_q, x_p - x_q \rangle + \beta \|g_p - g_q\|^2,
```

over the pairs ``(p, q) \in \{(i,j), (i,k), (j,k)\}``. Two ``7 \times 7``
matrices are built from these residuals together with nine free slack
[`Expression`](@ref)s (the two matrices differ by swapping the roles of the
two residual families), and both are constrained to be PSD through
[`PSDMatrix`](@ref) objects, following [1, Appendix F]. See the implementation
of `add_class_constraints!` in this file for the exact entries.

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1, "beta" => 1.0)
op = declare_function!(problem, CocoerciveStronglyMonotoneOperatorExpensive, param)
```

!!! note
    With `mu == 0` the class reduces to [`CocoerciveOperator`](@ref), and with
    `beta == 0` to [`StronglyMonotoneOperator`](@ref); the constructor emits a
    warning in those cases.

# Fields
- `mu::Float64`: strong monotonicity parameter ``\mu``.
- `beta::Float64`: cocoercivity parameter ``\beta``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
A constructive approach to strengthen algebraic descriptions of function and
operator classes.](https://arxiv.org/pdf/2504.14377.pdf)

[[2] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

See also [`declare_function!`](@ref), [`CocoerciveOperator`](@ref),
[`StronglyMonotoneOperator`](@ref), and
[`CocoerciveStronglyMonotoneOperatorCheap`](@ref).
"""
mutable struct CocoerciveStronglyMonotoneOperatorExpensive <: AbstractFunction
    mu::Float64
    beta::Float64
    _PEPit_func::PEPFunction

    function CocoerciveStronglyMonotoneOperatorExpensive(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); beta = float(param["beta"])
        mu == 0 && @warn "(PEPit) mu == 0; consider CocoerciveOperator instead."
        beta == 0 && @warn "(PEPit) beta == 0; consider StronglyMonotoneOperator instead."
        return new(mu, beta, func)
    end
end

add_constraint!(op::CocoerciveStronglyMonotoneOperatorExpensive, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::CocoerciveStronglyMonotoneOperatorExpensive) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::CocoerciveStronglyMonotoneOperatorExpensive)
    internal = op._PEPit_func
    pts = internal.list_of_points
    mu = op.mu; beta = op.beta

    build_T = function (xi, ti, xj, tj, xk, tk, opt)
        if opt == 1
            Aij = -(ti - tj) * (xi - xj) + mu * (xi - xj)^2
            Aik = -(ti - tk) * (xi - xk) + mu * (xi - xk)^2
            Ajk = -(tk - tj) * (xk - xj) + mu * (xk - xj)^2
            Bij = -(ti - tj) * (xi - xj) + beta * (ti - tj)^2
            Bik = -(ti - tk) * (xi - xk) + beta * (ti - tk)^2
            Bjk = -(tk - tj) * (xk - xj) + beta * (tk - tj)^2
        else
            Bij = -(ti - tj) * (xi - xj) + mu * (xi - xj)^2
            Bik = -(ti - tk) * (xi - xk) + mu * (xi - xk)^2
            Bjk = -(tk - tj) * (xk - xj) + mu * (xk - xj)^2
            Aij = -(ti - tj) * (xi - xj) + beta * (ti - tj)^2
            Aik = -(ti - tk) * (xi - xk) + beta * (ti - tk)^2
            Ajk = -(tk - tj) * (xk - xj) + beta * (tk - tj)^2
        end
        M14 = Expression(); M15 = Expression(); M16 = Expression(); M17 = Expression()
        M26 = Expression(); M27 = Expression(); M34 = Expression(); M37 = Expression(); M46 = Expression()
        M25 = -M14; M23 = -M15; M35 = -M16; M45 = -M27; M56 = -M37; M57 = -M46
        M55 = Aij - Ajk - Aik - 2 * (1 - 2 * beta * mu) * Bij - 2 * M17 - 2 * M26 - 2 * M34
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

_get_pep_func(op::CocoerciveStronglyMonotoneOperatorExpensive) = op._PEPit_func
