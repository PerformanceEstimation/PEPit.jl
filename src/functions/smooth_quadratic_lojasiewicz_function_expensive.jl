mutable struct SmoothQuadraticLojasiewiczFunctionExpensive <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function SmoothQuadraticLojasiewiczFunctionExpensive(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        @assert 0 <= mu <= L
        return new(mu, L, func)
    end
end

gradient!(f::SmoothQuadraticLojasiewiczFunctionExpensive, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothQuadraticLojasiewiczFunctionExpensive, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::SmoothQuadraticLojasiewiczFunctionExpensive) = stationary_point!(f._PEPit_func)
add_constraint!(func::SmoothQuadraticLojasiewiczFunctionExpensive, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::SmoothQuadraticLojasiewiczFunctionExpensive)
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


    _, _, fs = stationary[1]
    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        A = -fi + fj + 1 / 2 * (gi + gj) * (xi - xj) + 1 / (4 * L) * (gi - gj)^2 - L / 4 * (xi - xj)^2
        B = (L + mu) * (fi - fs - 1 / (2 * L) * gi^2)
        C = (L - mu) * (fj - fs + 1 / (2 * L) * gj^2)

        Mt11 = -A * (2 * L + mu)
        Mt12 = Expression()
        Mt22 = Expression()

        D = B - C - (L + 3 * mu) * A
        M11 = Mt11 - 4 * mu / (2 * L + mu) * Mt12 - D
        M12 = Mt12 - mu / (2 * L + mu) * Mt22 - (L + mu) / 2 * A + B
        M22 = Mt22 - B

        T1 = Matrix{Expression}(undef, 2, 2)
        T1[1, 1] = M11; T1[1, 2] = M12; T1[2, 1] = M12; T1[2, 2] = M22
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T1))

        T2 = Matrix{Expression}(undef, 2, 2)
        T2[1, 1] = Mt11; T2[1, 2] = Mt12; T2[2, 1] = Mt12; T2[2, 2] = Mt22
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T2))
    end
end

_get_pep_func(f::SmoothQuadraticLojasiewiczFunctionExpensive) = f._PEPit_func
