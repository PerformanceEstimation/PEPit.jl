@doc raw"""
    LinearOperator(param; reuse_gradient=true)

Interpolation class of linear operators ``M`` with singular values bounded by
``L``.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used. The adjoint operator ``M^\ast`` is available as the
    field `T`: calling `gradient!(M.T, u)` evaluates ``M^\ast u``.

# Class parameters
- `param["L"]`: upper bound ``L`` on the singular values of the operator.

# Interpolation conditions
Denoting by ``(x_i, y_i)`` the oracle pairs of ``M`` (with ``y_i = M x_i``)
and by ``(u_j, v_j)`` the oracle pairs of the adjoint ``M^\ast`` (with
``v_j = M^\ast u_j``), the following constraints are added (see [1]):

```math
\langle x_i, v_j \rangle = \langle y_i, u_j \rangle
\qquad \text{for all pairs } (i, j),
```

together with the two PSD constraints ``T^{(1)} \succeq 0`` and
``T^{(2)} \succeq 0``, where

```math
T^{(1)}_{ij} = L^2 \langle x_i, x_j \rangle - \langle y_i, y_j \rangle, \qquad
T^{(2)}_{ij} = L^2 \langle u_i, u_j \rangle - \langle v_i, v_j \rangle.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)
M = declare_function!(problem, LinearOperator, param)
y = gradient!(M, x)    # evaluates M * x
v = gradient!(M.T, u)  # evaluates M' * u
```

!!! note
    Linear operators are necessarily continuous, hence `reuse_gradient` is set
    to `true`.

# Fields
- `L::Float64`: singular value bound ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.
- `T::PEPFunction`: the adjoint linear operator ``M^\ast`` (created automatically by the constructor).

# References

[[1] N. Bousselmi, J. Hendrickx, F. Glineur (2023).
Interpolation Conditions for Linear Operators and applications to Performance
Estimation Problems. arXiv preprint.](https://arxiv.org/pdf/2302.08781.pdf)

See also [`declare_function!`](@ref), [`SymmetricLinearOperator`](@ref), and
[`SkewSymmetricLinearOperator`](@ref).
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
