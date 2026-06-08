@doc raw"""
    SmoothQuadraticLojasiewiczFunctionCheap(param; reuse_gradient=true)

Represent the `SmoothQuadraticLojasiewiczFunctionCheap` interpolation class in PEPit.jl.

Implement some constraints (which are not necessary and sufficient for interpolation)
for the class of smooth (not necessarily convex) functions that also satisfy a quadratic Lojasiewicz inequality
(sometimes also referred to as a Polyak-Lojasiewicz inequality). Extensive descriptions of such classes of
functions can be found in [1, 2].

The conditions implemented here are presented in [4, Proposition 3.2] (for alpha to be chosen)
and [4, Proposition 3.4] with smoothness conditions from [3].

# Warning

    Smooth functions satisfying a Lojasiewicz property do not enjoy known interpolation conditions.
    The conditions implemented in this class are necessary but a priori not sufficient for interpolation.
    Hence, the numerical results obtained when using this class might be non-tight upper bounds.

# Class parameters
- `L`: smoothness parameter
- `mu`: quadratic Lojasiewicz parameter
- `alpha`: relaxation parameter (in [0,2*mu/(2*L+mu)])

# Julia usage
```julia
problem = PEP()
param = OrderedDict("L" => 1.0)  # adapt keys to the class
f = declare_function!(problem, SmoothQuadraticLojasiewiczFunctionCheap, param)
```

# Fields
- `mu`: class parameter or auxiliary state stored as `Float64`.
- `L`: class parameter or auxiliary state stored as `Float64`.
- `alpha`: class parameter or auxiliary state stored as `Union{Float64,Nothing}`.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# Implementation
The constructor receives parameters through an `OrderedDict`; `add_class_constraints!` adds the interpolation model when [`solve!`](@ref) builds the SDP.
"""
mutable struct SmoothQuadraticLojasiewiczFunctionCheap <: AbstractFunction
    mu::Float64
    L::Float64
    alpha::Union{Float64,Nothing}
    _PEPit_func::PEPFunction

    function SmoothQuadraticLojasiewiczFunctionCheap(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        alpha = haskey(param, "alpha") && param["alpha"] !== nothing ? float(param["alpha"]) : nothing
        @assert 0 <= mu <= L
        if alpha !== nothing
            @assert 0 <= alpha <= 2 * mu / (2 * L + mu)
        end
        return new(mu, L, alpha, func)
    end
end

gradient!(f::SmoothQuadraticLojasiewiczFunctionCheap, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothQuadraticLojasiewiczFunctionCheap, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothQuadraticLojasiewiczFunctionCheap) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothQuadraticLojasiewiczFunctionCheap, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothQuadraticLojasiewiczFunctionCheap)
    internal = func._PEPit_func
    if isempty(internal.list_of_stationary_points)
        stationary_point!(internal)
    end
    points = internal.list_of_points
    stationary = internal.list_of_stationary_points
    L = func.L; mu = func.mu


    for (xi, gi, fi) in points, (xj, gj, fj) in stationary
        xi == xj && continue
        add_constraint!(func, fi - fj <= 1 / (2 * mu) * gi^2)
    end


    for (xi, gi, fi) in points, (xj, gj, fj) in stationary
        xi == xj && continue
        add_constraint!(func, fi - fj >= 1 / (2 * L) * gi^2)
    end


    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        add_constraint!(func, fi - fj >= 1 / 2 * (gi + gj) * (xi - xj) + 1 / (4 * L) * (gi - gj)^2 - L / 4 * (xi - xj)^2)
    end


    if func.alpha !== nothing
        alpha = func.alpha
        _, _, fs = stationary[1]
        const_coef = alpha / (1 - alpha) / (2 * mu - (L + mu) * alpha)
        for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
            i == j && continue
            add_constraint!(func, fi - fj >= 1 / 2 * (gi + gj) * (xi - xj) + 1 / (4 * L) * (gj - gi)^2 - L / 4 * (xj - xi)^2 +
                                  const_coef * ((1 - alpha)^2 * (L + mu) * (fi - fs - gi^2 / 2 / L) - (L - mu) * (fj - fs + gj^2 / 2 / L)))
        end
    end
end

_get_pep_func(f::SmoothQuadraticLojasiewiczFunctionCheap) = f._PEPit_func
