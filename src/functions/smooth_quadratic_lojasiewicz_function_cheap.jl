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
