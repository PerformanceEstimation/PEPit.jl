function proximal_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real)

    gx = Point()
    fx = Expression()


    x = x0 - gamma * gx


    add_point!(_get_pep_func(f), (x, gx, fx))

    return x, gx, fx
end
