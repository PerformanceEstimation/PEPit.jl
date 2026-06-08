@doc raw"""
    CocoerciveOperator(param; reuse_gradient=true)

Represent the `CocoerciveOperator` interpolation class in PEPit.jl.

Implement the interpolation constraints of the class of cocoercive (and maximally monotone) operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Class parameters
- `beta`: cocoercivity parameter

Cocoercive operators are characterized by the parameter $\beta$, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, CocoerciveOperator, param)
```

# Fields
- `beta`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
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
