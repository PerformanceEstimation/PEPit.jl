@doc raw"""
    StronglyMonotoneOperator(param; reuse_gradient=false)

Represent the `StronglyMonotoneOperator` interpolation class in PEPit.jl.

Implement interpolation constraints of the class of strongly monotone
(maximally monotone) operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Class parameters
- `mu`: strong monotonicity parameter

Strongly monotone (and maximally monotone) operators are characterized by the parameter $\mu$,
hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, StronglyMonotoneOperator, param)
```

# Fields
- `mu`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
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
