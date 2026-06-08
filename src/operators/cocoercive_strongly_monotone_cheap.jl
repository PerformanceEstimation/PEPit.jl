@doc raw"""
    CocoerciveStronglyMonotoneOperatorCheap(param; reuse_gradient=true)

Represent the `CocoerciveStronglyMonotoneOperatorCheap` interpolation class in PEPit.jl.

Implement some necessary constraints verified by the class of cocoercive
and strongly monotone (maximally) operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Class parameters
- `mu`: strong monotonicity parameter
- `beta`: cocoercivity parameter

Cocoercive operators are characterized by the parameters $\mu$ and $\beta$,
hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, CocoerciveStronglyMonotoneOperatorCheap, param)
```

# Fields
- `mu`: class parameter or auxiliary state stored as `Float64`.
- `beta`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
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
