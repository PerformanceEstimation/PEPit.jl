@doc raw"""
    LipschitzStronglyMonotoneOperatorCheap(param; reuse_gradient=true)

Represent the `LipschitzStronglyMonotoneOperatorCheap` interpolation class in PEPit.jl.

Implement some constraints (which are not necessary and sufficient for interpolation)
for the class of Lipschitz continuous strongly monotone (and maximally monotone) operators.

# Warning

    Lipschitz strongly monotone operators do not enjoy known interpolation conditions. The conditions implemented
    in this class are necessary but a priori not sufficient for interpolation. Hence, the numerical results
    obtained when using this class might be non-tight upper bounds (see Discussions in [1, Section 2]).

# Class parameters
- `mu`: strong monotonicity parameter
- `L`: Lipschitz parameter

Lipschitz continuous strongly monotone operators are characterized by parameters $\mu$ and `L`,
hence can be instantiated as

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, LipschitzStronglyMonotoneOperatorCheap, param)
```

# Fields
- `mu`: class parameter or auxiliary state stored as `Float64`.
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
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
