@doc raw"""
    SkewSymmetricLinearOperator(param; reuse_gradient=true)

Interpolation class of skew-symmetric linear operators ``M = -M^\ast`` with
singular values bounded by ``L``.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["L"]`: upper bound ``L`` on the singular values of the operator.

# Interpolation conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where
``g_i = M x_i`` denotes the operator value at ``x_i``, the following
skew-symmetry constraints are added (see [1]):

```math
\begin{aligned}
\langle x_i, g_j \rangle & = -\langle x_j, g_i \rangle && \text{for all } i < j, \\
\langle x_i, g_i \rangle & = 0 && \text{for all } i,
\end{aligned}
```

together with the PSD constraint ``T \succeq 0``, where

```math
T_{ij} = L^2 \langle x_i, x_j \rangle - \langle g_i, g_j \rangle.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)
M = declare_function!(problem, SkewSymmetricLinearOperator, param)
```

!!! note
    Skew-symmetric linear operators are necessarily continuous, hence
    `reuse_gradient` is set to `true`.

# Fields
- `L::Float64`: singular value bound ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] N. Bousselmi, J. Hendrickx, F. Glineur (2023).
Interpolation Conditions for Linear Operators and applications to Performance
Estimation Problems. arXiv preprint.](https://arxiv.org/pdf/2302.08781.pdf)

See also [`declare_function!`](@ref), [`LinearOperator`](@ref), and
[`SymmetricLinearOperator`](@ref).
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
