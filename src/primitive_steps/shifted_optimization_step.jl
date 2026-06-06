function shifted_optimization_step!(dir::AbstractPoint, f::AbstractFunction)

    x = Point()
    gx = dir
    fx = Expression()


    add_point!(_get_pep_func(f), (x, gx, fx))


    return x, gx, fx
end
