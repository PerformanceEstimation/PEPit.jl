mutable struct LipschitzStronglyMonotoneOperatorExpensive <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function LipschitzStronglyMonotoneOperatorExpensive(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        mu = float(param["mu"]); L = float(param["L"])
        L == Inf && @warn "(PEPit) L == Inf; consider StronglyMonotoneOperator instead."
        return new(mu, L, func)
    end
end

add_constraint!(op::LipschitzStronglyMonotoneOperatorExpensive, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)
fixed_point!(op::LipschitzStronglyMonotoneOperatorExpensive) = fixed_point!(op._PEPit_func)

function add_class_constraints!(op::LipschitzStronglyMonotoneOperatorExpensive)
    internal = op._PEPit_func
    pts = internal.list_of_points
    L = op.L; mu = op.mu

    build_T = function (xi, ti, xj, tj, xk, tk, opt)
        if opt == 1
            Aij = (ti - tj)^2 - L^2 * (xi - xj)^2
            Aik = (ti - tk)^2 - L^2 * (xi - xk)^2
            Ajk = (tk - tj)^2 - L^2 * (xk - xj)^2
            Bij = 2 * L * (-(ti - tj) * (xi - xj) + mu * (xi - xj)^2)
            Bik = 2 * L * (-(ti - tk) * (xi - xk) + mu * (xi - xk)^2)
            Bjk = 2 * L * (-(tk - tj) * (xk - xj) + mu * (xk - xj)^2)
        else
            Bij = (ti - tj)^2 - L^2 * (xi - xj)^2
            Bik = (ti - tk)^2 - L^2 * (xi - xk)^2
            Bjk = (tk - tj)^2 - L^2 * (xk - xj)^2
            Aij = 2 * L * (-(ti - tj) * (xi - xj) + mu * (xi - xj)^2)
            Aik = 2 * L * (-(ti - tk) * (xi - xk) + mu * (xi - xk)^2)
            Ajk = 2 * L * (-(tk - tj) * (xk - xj) + mu * (xk - xj)^2)
        end
        M14 = Expression(); M15 = Expression(); M16 = Expression(); M17 = Expression()
        M26 = Expression(); M27 = Expression(); M34 = Expression(); M37 = Expression(); M46 = Expression()
        M25 = -M14; M23 = -M15; M35 = -M16; M45 = -M27; M56 = -M37; M57 = -M46
        M55 = Aij + 2 * mu * Bij - Ajk - Aik - 2 * M17 - 2 * M26 - 2 * M34
        z = Expression(0.0)
        T = Matrix{Expression}(undef, 7, 7)
        T[1, :] = [-Bij, z, z, M14, M15, M16, M17]
        T[2, :] = [z, -Ajk, M23, z, M25, M26, M27]
        T[3, :] = [z, M23, -Bij, M34, M35, z, M37]
        T[4, :] = [M14, z, M34, -Bjk, M45, M46, z]
        T[5, :] = [M15, M25, M35, M45, M55, M56, M57]
        T[6, :] = [M16, M26, z, M46, M56, -Aik, z]
        T[7, :] = [M17, M27, M37, z, M57, z, -Bik]
        return T
    end

    n = length(pts)
    for i in 1:n, j in 1:n, k in 1:n
        (i == j && i == k) && continue
        xi, ti, _ = pts[i]; xj, tj, _ = pts[j]; xk, tk, _ = pts[k]
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=build_T(xi, ti, xj, tj, xk, tk, 1)))
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=build_T(xi, ti, xj, tj, xk, tk, 0)))
    end
end

_get_pep_func(op::LipschitzStronglyMonotoneOperatorExpensive) = op._PEPit_func
