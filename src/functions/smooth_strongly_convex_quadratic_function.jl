mutable struct SmoothStronglyConvexQuadraticFunction <: AbstractFunction
    mu::Float64
    L::Float64
    _PEPit_func::PEPFunction

    function SmoothStronglyConvexQuadraticFunction(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        stationary_point!(func)
        return new(float(param["mu"]), float(param["L"]), func)
    end
end

gradient!(f::SmoothStronglyConvexQuadraticFunction, p::Point) = gradient!(f._PEPit_func, p)
value!(f::SmoothStronglyConvexQuadraticFunction, p::Point) = value!(f._PEPit_func, p)
add_constraint!(func::SmoothStronglyConvexQuadraticFunction, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)


stationary_point!(f::SmoothStronglyConvexQuadraticFunction) = f._PEPit_func.list_of_stationary_points[1][1]

function add_class_constraints!(func::SmoothStronglyConvexQuadraticFunction)
    internal = func._PEPit_func
    points = internal.list_of_points
    xs, _, fs = internal.list_of_stationary_points[1]


    for (xi, gi, fi) in points
        add_constraint!(func, fi - fs == 0.5 * (xi - xs) * gi)
    end


    n = length(points)
    for i in 1:n, j in (i + 1):n
        xi, gi, _ = points[i]
        xj, gj, _ = points[j]
        add_constraint!(func, (xi - xs) * gj == (xj - xs) * gi)
    end


    if n > 0
        T = Matrix{Expression}(undef, n, n)
        for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
            T[i, j] = (func.L + func.mu) * (gi * (xj - xs)) - gi * gj - (func.mu * func.L) * ((xi - xs) * (xj - xs))
        end
        push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T))
    end
end

_get_pep_func(f::SmoothStronglyConvexQuadraticFunction) = f._PEPit_func
