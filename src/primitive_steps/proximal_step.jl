@doc raw"""
    proximal_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real)

Create the symbolic primitive step `proximal_step!`.

This routine performs a proximal step of step-size **gamma**, starting from **x0**, and on function **f**.
That is, it performs:

```math
\begin{aligned}
    x \triangleq \text{prox}_{\gamma f}(x_0) & \triangleq & \arg\min_x \left\{ \gamma f(x) + \frac{1}{2} \|x - x_0\|^2 \right\}, \\
    & \Updownarrow & \\
    0 & = & \gamma g_x + x - x_0 \text{ for some } g_x\in\partial f(x),\\
    & \Updownarrow & \\
    x & = & x_0 - \gamma g_x \text{ for some } g_x\in\partial f(x).
\end{aligned}
```

# Arguments
- `x0`: starting point x0.
- `f`: function on which the proximal step is computed.
- `gamma`: step-size parameter.

# Returns
- `x`: proximal point.
- `gx`: the (sub)gradient of f at x.
- `fx`: the function value of f on x.

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
function proximal_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real)

    gx = Point()
    fx = Expression()


    x = x0 - gamma * gx


    add_point!(_get_pep_func(f), (x, gx, fx))

    return x, gx, fx
end
