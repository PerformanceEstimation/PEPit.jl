@doc raw"""
    LinearOperator(param; reuse_gradient=true)

Represent the `LinearOperator` interpolation class in PEPit.jl.

Implement the interpolation constraints of the class of linear operators.

# Note

    Operator values can be requested through `gradient`, and `function values` should not be used.

# Class parameters
- `L`: singular values upper bound
- `T`: the adjunct linear operator

Linear operators are characterized by the parameter $L$, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, LinearOperator, param)
```

# Fields
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.
- `T`: class parameter or auxiliary state stored as `PEPFunction`.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct LinearOperator <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction
    T::PEPFunction

    function LinearOperator(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=true)
        L = param["L"]


        T = PEPFunction(is_leaf=true, reuse_gradient=true)
        T.counter = nothing
        Function_counter[] -= 1

        return new(L, func, T)
    end
end

gradient!(op::LinearOperator, p::Point) = gradient!(op._PEPit_func, p)
value!(op::LinearOperator, p::Point) = value!(op._PEPit_func, p)
stationary_point!(op::LinearOperator) = stationary_point!(op._PEPit_func)
add_constraint!(op::LinearOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)

function add_class_constraints!(op::LinearOperator)

    for (xi, yi, _) in op._PEPit_func.list_of_points
        for (uj, vj, _) in op.T.list_of_points
            add_constraint!(op, xi * vj == yi * uj)
        end
    end


    N1 = length(op._PEPit_func.list_of_points)
    if N1 > 0
        T1 = Matrix{Expression}(undef, N1, N1)
        for (i, (xi, yi, _)) in enumerate(op._PEPit_func.list_of_points)
            for (j, (xj, yj, _)) in enumerate(op._PEPit_func.list_of_points)
                T1[i, j] = op.L^2 * xi * xj - yi * yj
            end
        end
        push!(op._PEPit_func.list_of_class_psd, PSDMatrix(matrix_of_expressions=T1))
    end


    N2 = length(op.T.list_of_points)
    if N2 > 0
        T2 = Matrix{Expression}(undef, N2, N2)
        for (i, (ui, vi, _)) in enumerate(op.T.list_of_points)
            for (j, (uj, vj, _)) in enumerate(op.T.list_of_points)
                T2[i, j] = op.L^2 * ui * uj - vi * vj
            end
        end
        push!(op._PEPit_func.list_of_class_psd, PSDMatrix(matrix_of_expressions=T2))
    end
end

_get_pep_func(op::LinearOperator) = op._PEPit_func
