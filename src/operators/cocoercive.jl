@doc raw"""
    CocoerciveOperator(param; reuse_gradient=true)

Interpolation class of ``\beta``-cocoercive (and maximally monotone)
operators.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["beta"]`: cocoercivity parameter ``\beta``.

# Interpolation conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where ``g_i``
denotes the operator value at ``x_i``, the following constraint is added for
every pair ``i \neq j`` (see [1]):

```math
\langle g_i - g_j, x_i - x_j \rangle \geqslant \beta \|g_i - g_j\|^2.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("beta" => 1.0)
op = declare_function!(problem, CocoerciveOperator, param)
```

!!! note
    Cocoercive operators are necessarily continuous, hence `reuse_gradient` is
    set to `true`. With `beta == 0` the class reduces to monotone operators;
    the constructor emits a warning suggesting [`MonotoneOperator`](@ref) in
    that case.

# Fields
- `beta::Float64`: cocoercivity parameter ``\beta``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

See also [`declare_function!`](@ref), [`MonotoneOperator`](@ref), and
[`CocoerciveStronglyMonotoneOperatorCheap`](@ref).
"""
mutable struct CocoerciveOperator <: AbstractFunction
    beta::Float64
    _PEPit_func::PEPFunction

    function CocoerciveOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        beta = float(param["beta"])
        beta == 0 && @warn "(PEPit) beta == 0 reduces to a monotone operator; consider MonotoneOperator instead."
        return new(beta, func)
    end
end

add_constraint!(op::CocoerciveOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::CocoerciveOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::CocoerciveOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) - op.beta * (gi - gj)^2 >= 0)
    end
end

_get_pep_func(op::CocoerciveOperator) = op._PEPit_func
