@doc raw"""
    NegativelyComonotoneOperator(param; reuse_gradient=true)

Class of ``\rho``-negatively comonotone operators, modeled through necessary
constraints (see, e.g., [1] for a discussion of this class of nonmonotone
operators).

Overrides `add_class_constraints!` to add the conditions of the class when
[`solve!`](@ref) builds the SDP.

!!! warning
    Those constraints might not be sufficient, thus the characterized class
    might contain more operators.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["rho"]`: comonotonicity parameter ``\rho`` (``> 0``).

# Necessary conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where ``g_i``
denotes the operator value at ``x_i``, the following constraint is added for
every pair ``i \neq j``:

```math
\langle g_i - g_j, x_i - x_j \rangle \geqslant -\rho \|g_i - g_j\|^2.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("rho" => 0.1)
op = declare_function!(problem, NegativelyComonotoneOperator, param)
```

!!! note
    With `rho == 0` the class reduces to monotone operators; the constructor
    emits a warning suggesting [`MonotoneOperator`](@ref) in that case.

# Fields
- `rho::Float64`: comonotonicity parameter ``\rho``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] E. Gorbunov, A. Taylor, S. Horváth, G. Gidel (2023).
Convergence of proximal point and extragradient-based methods beyond
monotonicity: the case of negative comonotonicity. International Conference on
Machine Learning.](https://proceedings.mlr.press/v202/gorbunov23a/gorbunov23a.pdf)

See also [`declare_function!`](@ref), [`MonotoneOperator`](@ref), and
[`CocoerciveOperator`](@ref).
"""
mutable struct NegativelyComonotoneOperator <: AbstractFunction
    rho::Float64
    _PEPit_func::PEPFunction

    function NegativelyComonotoneOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        rho = float(param["rho"])
        rho < 0 && @warn "(PEPit) The parameter rho is expected to be positive."
        rho == 0 && @warn "(PEPit) rho == 0 reduces to a monotone operator; consider MonotoneOperator instead."
        return new(rho, func)
    end
end

add_constraint!(op::NegativelyComonotoneOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::NegativelyComonotoneOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::NegativelyComonotoneOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) + op.rho * (gi - gj)^2 >= 0)
    end
end

_get_pep_func(op::NegativelyComonotoneOperator) = op._PEPit_func
