@doc raw"""
    MonotoneOperator(param=OrderedDict(); reuse_gradient=false)

Interpolation class of maximally monotone operators (see, e.g., [1] for an
extensive discussion of maximal monotonicity).

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

!!! note
    Operator values are requested through [`gradient!`](@ref); function values
    should not be used.

# Class parameters
General maximally monotone operators are not characterized by any parameter,
so `param` may be left empty.

# Interpolation conditions
Associating with each oracle call ``i`` the pair ``(x_i, g_i)``, where ``g_i``
denotes the operator value at ``x_i``, the following constraint is added for
every pair ``i \neq j``:

```math
\langle g_i - g_j, x_i - x_j \rangle \geqslant 0.
```

Maximality guarantees that any monotone set of pairs can be extended to a
maximally monotone operator, so these conditions are interpolation conditions
for the class.

# Julia usage
```julia
problem = PEP()
op = declare_function!(problem, MonotoneOperator, OrderedDict())
```

# Fields
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] H. H. Bauschke and P. L. Combettes (2017).
Convex Analysis and Monotone Operator Theory in Hilbert Spaces.
Springer New York.](https://link.springer.com/book/10.1007/978-3-319-48311-5)

See also [`declare_function!`](@ref), [`StronglyMonotoneOperator`](@ref), and
[`CocoerciveOperator`](@ref).
"""
mutable struct MonotoneOperator <: AbstractFunction
    _PEPit_func::PEPFunction

    function MonotoneOperator(param=OrderedDict(); is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(func)
    end
end

add_constraint!(op::MonotoneOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)

function add_class_constraints!(op::MonotoneOperator)
    pts = op._PEPit_func.list_of_points
    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj) * (xi - xj) >= 0)
    end
end

_get_pep_func(op::MonotoneOperator) = op._PEPit_func
