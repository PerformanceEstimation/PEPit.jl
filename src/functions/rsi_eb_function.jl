@doc raw"""
    RsiEbFunction(param; reuse_gradient=false)

Represent the `RsiEbFunction` interpolation class in PEPit.jl.

Implement the interpolation constraints of the class of functions verifying
the "lower" restricted secant inequality ($\text{RSI}^-$) and the "upper" error bound ($\text{EB}^+$).

# Class parameters
- `mu`: Restricted sequent inequality parameter
- `L`: Error bound parameter
$\text{RSI}^-$ and $\text{EB}^+$ functions are characterized by parameters $\mu$ and `L`,
hence can be instantiated as

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, RsiEbFunction, param)
```

# Fields
- `mu`: class parameter or auxiliary state stored as `Float64`.
- `L`: class parameter or auxiliary state stored as `Float64`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct RsiEbFunction <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function RsiEbFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=false)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        return new(float(param["mu"]), float(param["L"]), func)
    end
end

gradient!(f::RsiEbFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::RsiEbFunction, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::RsiEbFunction) = stationary_point!(f._PEPit_func)
add_constraint!(func::RsiEbFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::RsiEbFunction)
    internal = func._PEPit_func
    if isempty(internal.list_of_stationary_points)
        stationary_point!(internal)
    end
    points = internal.list_of_points
    stationary = internal.list_of_stationary_points


    for (xi, gi, fi) in stationary, (xj, gj, fj) in points
        xi == xj && continue
        add_constraint!(func, (gi - gj) * (xi - xj) - func.mu * (xi - xj)^2 >= 0)
    end


    for (xi, gi, fi) in stationary, (xj, gj, fj) in points
        xi == xj && continue
        add_constraint!(func, (gi - gj)^2 - func.L^2 * (xi - xj)^2 <= 0)
    end
end

_get_pep_func(f::RsiEbFunction) = f._PEPit_func
