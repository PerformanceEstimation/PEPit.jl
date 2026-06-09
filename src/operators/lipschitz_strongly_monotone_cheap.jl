@doc raw"""
    LipschitzStronglyMonotoneOperatorCheap(param; reuse_gradient=true)

Class of ``L``-Lipschitz continuous and ``\mu``-strongly monotone (maximally
monotone) operators, modeled through a cheap set of necessary constraints.

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
denotes the operator value at ``x_i``, the following constraints are added for
every pair ``i \neq j`` (see [1]):

```math
\begin{aligned}
\langle g_i - g_j, x_i - x_j \rangle & \geqslant \mu \|x_i - x_j\|^2, \\
\|g_i - g_j\|^2 & \leqslant L^2 \|x_i - x_j\|^2.
\end{aligned}
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1, "L" => 1.0)
op = declare_function!(problem, LipschitzStronglyMonotoneOperatorCheap, param)
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

[[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

See also [`declare_function!`](@ref), [`LipschitzOperator`](@ref),
[`StronglyMonotoneOperator`](@ref), and
[`LipschitzStronglyMonotoneOperatorExpensive`](@ref).
"""
mutable struct LipschitzStronglyMonotoneOperatorCheap <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function LipschitzStronglyMonotoneOperatorCheap(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        L == Inf && @warn "(PEPit) L == Inf; consider StronglyMonotoneOperator instead."
        return new(mu, L, func)
    end
end

add_constraint!(op::LipschitzStronglyMonotoneOperatorCheap, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::LipschitzStronglyMonotoneOperatorCheap) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::LipschitzStronglyMonotoneOperatorCheap)
    pts = op._PEPit_func.list_of_points

    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) - op.mu * (xi - xj)^2 >= 0)
    end

    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj)^2 - op.L^2 * (xi - xj)^2 <= 0)
    end
end

_get_pep_func(op::LipschitzStronglyMonotoneOperatorCheap) = op._PEPit_func
