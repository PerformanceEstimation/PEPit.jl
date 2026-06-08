@doc raw"""
    LipschitzOperator(param; reuse_gradient=true)

Represent the `LipschitzOperator` interpolation class in PEPit.jl.

Implement the interpolation constraints of the class of Lipschitz continuous operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Class parameters
- `L`: Lipschitz parameter

Cocoercive operators are characterized by the parameter $L$, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, LipschitzOperator, param)
```

# Fields
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct LipschitzOperator <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function LipschitzOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=true)
        L = param["L"]
        if L == Inf
            @warn "(PEPit) The class of L-Lipschitz operators with L == Inf implies no constraint: it contains all multi-valued mappings."
        end
        return new(L, func)
    end
end

add_constraint!(op::LipschitzOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::LipschitzOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::LipschitzOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        constraint = (gi - gj)^2 - op.L^2 * (xi - xj)^2 <= 0
        add_constraint!(op, constraint)
    end
end

_get_pep_func(op::LipschitzOperator) = op._PEPit_func
