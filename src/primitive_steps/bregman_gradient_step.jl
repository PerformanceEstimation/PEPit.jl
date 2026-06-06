function bregman_gradient_step!(gx0::AbstractPoint, sx0::AbstractPoint, mirror_map::AbstractFunction, gamma::Real)

    x = Point()
    hx = Expression()


    sx = sx0 - gamma * gx0


    add_point!(_get_pep_func(mirror_map), (x, sx, hx))


    return x, sx, hx
end
