@doc raw"""
    StronglyMonotoneOperator(param; reuse_gradient=false)

Interpolation class of ``\mu``-strongly monotone (and maximally monotone)
operators.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["mu"]`: strong monotonicity parameter ``\mu``.

# Interpolation conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where ``g_i``
denotes the operator value at ``x_i``, the following constraint is added for
every pair ``i \neq j``:

```math
\langle g_i - g_j, x_i - x_j \rangle \geqslant \mu \|x_i - x_j\|^2.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1)
op = declare_function!(problem, StronglyMonotoneOperator, param)
```

# Fields
- `mu::Float64`: strong monotonicity parameter ``\mu``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

Discussions and appropriate pointers for the problem of interpolation of
maximally monotone operators can be found in:

[[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

See also [`declare_function!`](@ref), [`MonotoneOperator`](@ref), and
[`LipschitzStronglyMonotoneOperatorCheap`](@ref).
"""
mutable struct StronglyMonotoneOperator <: AbstractFunction
    mu::Float64
    _PEPit_func::PEPFunction

    function StronglyMonotoneOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["mu"]), func)
    end
end

add_constraint!(op::StronglyMonotoneOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::StronglyMonotoneOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::StronglyMonotoneOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) - op.mu * (xi - xj)^2 >= 0)
    end
end

_get_pep_func(op::StronglyMonotoneOperator) = op._PEPit_func
