@doc raw"""
    MonotoneOperator(param=OrderedDict(); reuse_gradient=false)

Represent the `MonotoneOperator` interpolation class in PEPit.jl.

Implement interpolation constraints for the class of maximally monotone operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

General maximally monotone operators are not characterized by any parameter, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, MonotoneOperator, param)
```

# Fields
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct MonotoneOperator <: AbstractFunction
    _PEPit_func::PEPFunction

    function MonotoneOperator(param=OrderedDict(); is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(func)
    end
end

add_constraint!(op::MonotoneOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)

function add_class_constraints!(op::MonotoneOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) >= 0)
    end
end

_get_pep_func(op::MonotoneOperator) = op._PEPit_func
