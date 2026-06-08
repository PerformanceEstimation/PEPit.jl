@doc raw"""
    SkewSymmetricLinearOperator(param; reuse_gradient=true)

Represent the `SkewSymmetricLinearOperator` interpolation class in PEPit.jl.

Implement the interpolation constraints for the class of skew-symmetric linear operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Class parameters
- `L`: singular values upper bound

Skew-Symmetric Linear operators are characterized by parameters $L$, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, SkewSymmetricLinearOperator, param)
```

# Fields
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct SkewSymmetricLinearOperator <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function SkewSymmetricLinearOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["L"]), func)
    end
end

add_constraint!(op::SkewSymmetricLinearOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::SkewSymmetricLinearOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::SkewSymmetricLinearOperator)
    internal = op._PEPit_func
    pts = internal.list_of_points


    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, xi * gj == -(xj * gi))
    end


    for (xi, gi, _) in pts
        add_constraint!(op, xi * gi == 0)
    end


    N = length(pts)
    if N > 0
        T = Matrix{Expression}(undef, N, N)
        for (i, (xi, gi, fi)) in enumerate(pts), (j, (xj, gj, fj)) in enumerate(pts)
            T[i, j] = (op.L^2) * (xi * xj) - gi * gj
        end
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T))
    end
end

_get_pep_func(op::SkewSymmetricLinearOperator) = op._PEPit_func
