@doc raw"""
    bregman_gradient_step!(gx0::AbstractPoint, sx0::AbstractPoint, mirror_map::AbstractFunction, gamma::Real)

Create the symbolic primitive step `bregman_gradient_step!`.

This routine outputs $x$ by performing a mirror step of step-size $\gamma$.
That is, denoting $f$ the function to be minimized
and $h$ the **mirror map**, it performs

```math
x = \arg\min_x \left[ f(x_0) + \left< \nabla f(x_0);\, x - x_0 \right>
```
          + \frac{1}{\gamma} D_h(x; x_0) \right],

where $D_h(x; x_0)$ denotes the Bregman divergence of $h$ on $x$ with respect to $x_0$.

```math
D_h(x; x_0) \triangleq h(x) - h(x_0) - \left< \nabla h(x_0);\, x - x_0 \right>.
```

# Arguments
- `gx0`: descent direction $\textbf{gx0} \triangleq \nabla f(x_0)$.
- `sx0`: starting gradient $\textbf{sx0} \triangleq \nabla h(x_0)$.
- `mirror_map`: the reference function $h$ we computed Bregman divergence of.
- `gamma`: step-size parameter.

# Returns
- `x`: new iterate $\textbf{x} \triangleq x$.
- `sx`: $h$'s gradient on new iterate $x$ $\textbf{sx} \triangleq \nabla h(x)$.
- `hx`: $h$'s value on new iterate $\textbf{hx} \triangleq h(x)$.

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
function bregman_gradient_step!(gx0::AbstractPoint, sx0::AbstractPoint, mirror_map::AbstractFunction, gamma::Real)

    x = Point()
    hx = Expression()


    sx = sx0 - gamma * gx0


    add_point!(_get_pep_func(mirror_map), (x, sx, hx))


    return x, sx, hx
end
