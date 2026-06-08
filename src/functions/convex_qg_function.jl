@doc raw"""
    ConvexQGFunction(param; reuse_gradient=false)

Represent the `ConvexQGFunction` interpolation class in PEPit.jl.

Implement the interpolation constraints of the class of quadratically upper bounded ($\text{QG}^+$ [1]),
i.e. $\forall x, f(x) - f_\star \leqslant \frac{L}{2} \|x-x_\star\|^2$, and convex functions.

# Class parameters
- `L`: The quadratic upper bound parameter

General quadratically upper bounded ($\text{QG}^+$) convex functions are characterized
by the quadratic growth parameter `L`, hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, ConvexQGFunction, param)
```

# Fields
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct ConvexQGFunction <: AbstractFunction
    L::Float64
    _PEPit_func::PEPFunction

    function ConvexQGFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["L"]), func)
    end
end

gradient!(f::ConvexQGFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::ConvexQGFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::ConvexQGFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::ConvexQGFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::ConvexQGFunction)
    internal = func._PEPit_func
    if isempty(internal.list_of_stationary_points)
        stationary_point!(internal)
    end
    points = internal.list_of_points
    stationary = internal.list_of_stationary_points


    for (xi, gi, fi) in stationary, (xj, gj, fj) in points
        xi == xj && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj) + 1 / (2 * func.L) * gj^2)
    end


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, fi - fj >= gj * (xi - xj))
    end
end

_get_pep_func(f::ConvexQGFunction) = f._PEPit_func
