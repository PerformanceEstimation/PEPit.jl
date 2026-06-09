@doc raw"""
    RsiEbFunction(param; reuse_gradient=false)

Interpolation class of functions verifying the "lower" restricted secant
inequality (``\text{RSI}^-``) and the "upper" error bound (``\text{EB}^+``).

Overrides `add_class_constraints!` to add the interpolation conditions of the
class when [`solve!`](@ref) builds the SDP.

# Class parameters
- `param["mu"]`: restricted secant inequality parameter ``\mu``.
- `param["L"]`: error bound parameter ``L``.

# Interpolation conditions
A stationary point ``(x_\star, g_\star = 0, f_\star)`` is created automatically
if none was requested through [`stationary_point!`](@ref). Associating with
each oracle call ``j`` the triplet ``(x_j, g_j, f_j)`` of point, subgradient,
and function value, the following constraints are added for every stationary
triplet and every ``j`` with ``x_j \neq x_\star``:

```math
\begin{aligned}
\langle g_\star - g_j, x_\star - x_j \rangle & \geqslant \mu \|x_\star - x_j\|^2
&& (\text{RSI}^-), \\
\|g_\star - g_j\|^2 & \leqslant L^2 \|x_\star - x_j\|^2 && (\text{EB}^+).
\end{aligned}
```

# Julia usage
```julia
problem = PEP()
param = OrderedDict("mu" => 0.1, "L" => 1.0)
f = declare_function!(problem, RsiEbFunction, param)
```

# Fields
- `mu::Float64`: restricted secant inequality parameter ``\mu``.
- `L::Float64`: error bound parameter ``L``.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

A definition of the class of ``\text{RSI}^-`` and ``\text{EB}^+`` functions can
be found in [1].

[[1] C. Guille-Escuret, B. Goujaud, A. Ibrahim, I. Mitliagkas (2022).
Gradient Descent Is Optimal Under Lower Restricted Secant Inequality And Upper
Error Bound. arXiv 2203.00342.](https://arxiv.org/pdf/2203.00342.pdf)

See also [`declare_function!`](@ref) and [`stationary_point!`](@ref).
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
