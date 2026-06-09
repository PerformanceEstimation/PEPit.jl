using OrderedCollections

@doc raw"""
    NonexpansiveOperator(param=OrderedDict(); reuse_gradient=true)

Interpolation class of (possibly inconsistent) nonexpansive operators.

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
- `param["v"]` (optional, default `nothing`): infimal displacement vector ``v``, given as a [`Point`](@ref).

Nonexpansive operators are not otherwise characterized by any parameter.
Omitting `"v"` corresponds to the consistent case.

# Interpolation conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where
``g_i = T(x_i)`` denotes the operator value at ``x_i``, the following
constraint is added for every pair ``i \neq j``:

```math
\|g_i - g_j\|^2 \leqslant \|x_i - x_j\|^2.
```

When the infimal displacement vector ``v`` is provided, the following
constraint is also added for every ``i`` (see [2]):

```math
\|v\|^2 \leqslant \langle x_i - g_i, v \rangle.
```

# Julia usage
```julia
problem = PEP()
op = declare_function!(problem, NonexpansiveOperator, OrderedDict())
```

!!! note
    Any nonexpansive operator ``T`` has a unique vector called the *infimal
    displacement vector*, which we denote by ``v``. If a nonexpansive operator
    is consistent, i.e., has a fixed point, then ``v = 0``. If ``v`` is
    nonzero, the operator is inconsistent, i.e., does not have a fixed point.

# Fields
- `v::Union{Point,Nothing}`: infimal displacement vector ``v``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

Discussions and appropriate pointers for the interpolation problem can be
found in:

[[1] E. Ryu, A. Taylor, C. Bergeling, P. Giselsson (2020).
Operator splitting performance estimation: Tight contraction factors and
optimal parameter selection. SIAM Journal on Optimization, 30(3),
2251-2271.](https://arxiv.org/pdf/1812.00146.pdf)

[[2] J. Park, E. Ryu (2023).
Accelerated Infeasibility Detection of Constrained Optimization and
Fixed-Point Iterations. arXiv preprint:2303.15876.](https://arxiv.org/pdf/2303.15876.pdf)

See also [`declare_function!`](@ref), [`LipschitzOperator`](@ref), and
[`fixed_point!`](@ref).
"""
mutable struct NonexpansiveOperator <: AbstractFunction
    v::Union{Point,Nothing}
    _PEPit_func::PEPFunction

    function NonexpansiveOperator(param=OrderedDict(); is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=true)
        v = haskey(param, "v") ? param["v"] : nothing
        return new(v, func)
    end
end

gradient!(op::NonexpansiveOperator, p::Point) = gradient!(op._PEPit_func, p)
value!(op::NonexpansiveOperator, p::Point) = value!(op._PEPit_func, p)
stationary_point!(op::NonexpansiveOperator) = stationary_point!(op._PEPit_func)
add_constraint!(op::NonexpansiveOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)

function add_class_constraints!(op::NonexpansiveOperator)
    pts = op._PEPit_func.list_of_points


    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj)^2 - (xi - xj)^2 <= 0)
    end


    if op.v !== nothing
        for (xi, gi, _) in pts
            add_constraint!(op, op.v^2 - (xi - gi) * op.v <= 0)
        end
    end
end

_get_pep_func(op::NonexpansiveOperator) = op._PEPit_func
