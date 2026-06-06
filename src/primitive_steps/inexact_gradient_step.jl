function inexact_gradient_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real, epsilon::Real; notion::String="absolute")

    gx0, fx0 = oracle!(f, x0)


    dx0 = Point()


    if notion == "absolute"
        add_constraint!(f, (gx0 - dx0)^2 - epsilon^2 <= 0)
    elseif notion == "relative"
        add_constraint!(f, (gx0 - dx0)^2 - epsilon^2 * gx0^2 <= 0)
    else
        error("inexact_gradient_step! supports only notion in [\"absolute\", \"relative\"], got $notion")
    end


    x = x0 - gamma * dx0


    return x, dx0, fx0
end
