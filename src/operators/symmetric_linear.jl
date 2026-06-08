@doc raw"""
    SymmetricLinearOperator(param; reuse_gradient=true)

Represent the `SymmetricLinearOperator` interpolation class in PEPit.jl.

Implement the interpolation constraints for the class of symmetric linear operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Class parameters
- `mu`: eigenvalues lower bound
- `L`: eigenvalues upper bound

Symmetric Linear operators are characterized by parameters $\mu$ and `L`,
hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, SymmetricLinearOperator, param)
```

# Fields
- `mu`: class parameter or auxiliary state stored as `Float64`.
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct SymmetricLinearOperator <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function SymmetricLinearOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["mu"]), float(param["L"]), func)
    end
end

add_constraint!(op::SymmetricLinearOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::SymmetricLinearOperator) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::SymmetricLinearOperator)
    internal = op._PEPit_func
    pts = internal.list_of_points


    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]; xj, gj, _ = pts[j]
        add_constraint!(op, xi * gj == xj * gi)
    end


    N = length(pts)
    if N > 0
        T = Matrix{Expression}(undef, N, N)
        for (i, (xi, gi, fi)) in enumerate(pts), (j, (xj, gj, fj)) in enumerate(pts)
            T[i, j] = op.L * gi * xj - gi * gj - (op.mu * op.L) * (xi * xj) + op.mu * (xi * gj)
        end
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T))
    end
end

_get_pep_func(op::SymmetricLinearOperator) = op._PEPit_func
