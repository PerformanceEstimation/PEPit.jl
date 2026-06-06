function linear_optimization_step!(dir::AbstractPoint, ind::AbstractFunction)

    x = Point()
    gx = -dir
    fx = Expression()


    add_point!(_get_pep_func(ind), (x, gx, fx))


    return x, gx, fx
end
