@doc raw"""
    LipschitzOperator(param; reuse_gradient=true)

Interpolation class of ``L``-Lipschitz continuous operators.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["L"]`: Lipschitz continuity parameter ``L``.

# Interpolation conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where ``g_i``
denotes the operator value at ``x_i``, the following constraint is added for
every pair ``i \neq j`` (see [1, 2, 3], or e.g. [4, Fact 2]):

```math
\|g_i - g_j\|^2 \leqslant L^2 \|x_i - x_j\|^2.
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)
op = declare_function!(problem, LipschitzOperator, param)
```

!!! note
    Setting ``L = 1`` models a nonexpansive operator, and ``L < 1`` a
    contracting operator. With `L == Inf` the class adds no constraint (it
    contains all multi-valued mappings); the constructor emits a warning in
    that case.

# Fields
- `L::Float64`: Lipschitz continuity parameter ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] M. Kirszbraun (1934).
Über die zusammenziehende und Lipschitzsche Transformationen.
Fundamenta Mathematicae, 22.](https://eudml.org/doc/212681)

[[2] F.A. Valentine (1943).
On the extension of a vector function so as to preserve a Lipschitz condition.
Bulletin of the American Mathematical Society, 49(2).](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-49/issue-2)

[[3] F.A. Valentine (1945).
A Lipschitz condition preserving extension for a vector function.
American Journal of Mathematics, 67(1).](https://www.jstor.org/stable/2371917)

Discussions and appropriate pointers for the interpolation problem can be
found in:

[[4] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

See also [`declare_function!`](@ref), [`NonexpansiveOperator`](@ref), and
[`LipschitzStronglyMonotoneOperatorCheap`](@ref).
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
