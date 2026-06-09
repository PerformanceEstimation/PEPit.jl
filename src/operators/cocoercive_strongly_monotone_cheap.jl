@doc raw"""
    CocoerciveStronglyMonotoneOperatorCheap(param; reuse_gradient=true)

Class of ``\beta``-cocoercive and ``\mu``-strongly monotone (maximally
monotone) operators, modeled through a cheap set of necessary constraints.

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
denotes the operator value at ``x_i``, the following constraints are added for
every pair ``i \neq j`` (see [1]):

```math
\begin{aligned}
\langle g_i - g_j, x_i - x_j \rangle & \geqslant \beta \|g_i - g_j\|^2, \\
\langle g_i - g_j, x_i - x_j \rangle & \geqslant \mu \|x_i - x_j\|^2.
\end{aligned}
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1, "beta" => 1.0)
op = declare_function!(problem, CocoerciveStronglyMonotoneOperatorCheap, param)
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

[[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

See also [`declare_function!`](@ref), [`CocoerciveOperator`](@ref),
[`StronglyMonotoneOperator`](@ref), and
[`CocoerciveStronglyMonotoneOperatorExpensive`](@ref).
"""
mutable struct CocoerciveStronglyMonotoneOperatorCheap <: AbstractFunction
    mu::Float64
    beta::Float64
    _PEPit_func::PEPFunction

    function CocoerciveStronglyMonotoneOperatorCheap(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); beta = float(param["beta"])
        mu == 0 && @warn "(PEPit) mu == 0; consider CocoerciveOperator instead."
        beta == 0 && @warn "(PEPit) beta == 0; consider StronglyMonotoneOperator instead."
        return new(mu, beta, func)
    end
end

add_constraint!(op::CocoerciveStronglyMonotoneOperatorCheap, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::CocoerciveStronglyMonotoneOperatorCheap) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::CocoerciveStronglyMonotoneOperatorCheap)
    pts = op._PEPit_func.list_of_points

    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) - op.beta * (gi - gj)^2 >= 0)
    end

    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) - op.mu * (xi - xj)^2 >= 0)
    end
end

_get_pep_func(op::CocoerciveStronglyMonotoneOperatorCheap) = op._PEPit_func
