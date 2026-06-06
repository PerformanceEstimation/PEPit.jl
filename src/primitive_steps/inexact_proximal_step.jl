function inexact_proximal_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real; opt::String="PD_gapII")
    if opt == "PD_gapI"

        v = Point()
        w = Point()
        fw = Expression()
        add_point!(_get_pep_func(f), (w, v, fw))

        x = Point()
        gx = Point()
        fx = Expression()
        add_point!(_get_pep_func(f), (x, gx, fx))

        eps_var = Expression()
        e = x - x0 + gamma * v
        eps_sub = fx - fw - v * (x - w)
        constraint = (e^2 / 2 + gamma * eps_sub <= eps_var)

    elseif opt == "PD_gapII"

        e = Point()
        gx = Point()
        x = x0 - gamma * gx + e
        fx = Expression()
        add_point!(_get_pep_func(f), (x, gx, fx))
        eps_var = Expression()
        constraint = (e^2 / 2 <= eps_var)
        w, v, fw = x, gx, fx

    elseif opt == "PD_gapIII"

        x, gx, w = Point(), Point(), Point()
        v = (x0 - x) / gamma
        fw, fx = Expression(), Expression()
        add_point!(_get_pep_func(f), (x, gx, fx))
        add_point!(_get_pep_func(f), (w, v, fw))
        eps_var = Expression()
        eps_sub = fx - fw - v * (x - w)
        constraint = (gamma * eps_sub <= eps_var)

    else
        error("inexact_proximal_step! supports only opt in ['PD_gapI','PD_gapII','PD_gapIII'], got $opt")
    end


    add_constraint!(f, constraint)

    return x, gx, fx, w, v, fw, eps_var
end
