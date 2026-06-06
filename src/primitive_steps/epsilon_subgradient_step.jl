function epsilon_subgradient_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real)

    g0 = Point()
    f0 = value!(f, x0)
    epsilon = Expression()


    x = x0 - gamma * g0


    y = Point()
    fy = Expression()
    add_point!(_get_pep_func(f), (y, g0, fy))
    fstarg0 = g0 * y - fy


    constraint = (f0 + fstarg0 - g0 * x0 <= epsilon)
    add_constraint!(f, constraint)


    return x, g0, f0, epsilon
end
