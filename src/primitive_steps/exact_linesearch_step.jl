function exact_linesearch_step!(x0::AbstractPoint, f::AbstractFunction, directions)

    x = Point()


    gx, fx = oracle!(f, x)


    add_constraint!(f, (x - x0) * gx == 0)
    for d in directions
        add_constraint!(f, d * gx == 0)
    end


    return x, gx, fx
end
