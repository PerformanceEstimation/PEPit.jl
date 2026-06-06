function bregman_proximal_step!(sx0::AbstractPoint, mirror_map::AbstractFunction, min_function::AbstractFunction, gamma::Real)

    x = Point()


    gx = Point()
    fx = Expression()


    sx = sx0 - gamma * gx
    hx = Expression()


    add_point!(_get_pep_func(min_function), (x, gx, fx))
    add_point!(_get_pep_func(mirror_map), (x, sx, hx))


    return x, sx, hx, gx, fx
end
