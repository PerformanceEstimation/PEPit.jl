@doc raw"""
    exact_linesearch_step!(x0::AbstractPoint, f::AbstractFunction, directions)

Create the symbolic primitive step `exact_linesearch_step!`.

This routine outputs some $x$ by *mimicking* an exact line/span search in specified directions.
It is used for instance by the Julia examples
`examples/unconstrained_convex_minimization/gradient_exact_line_search.jl` and
`examples/unconstrained_convex_minimization/conjugate_gradient.jl`.

The routine aims at mimicking the operation:

```math
\begin{aligned}
    x & = & x_0 - \sum_{i=1}^{T} \gamma_i d_i,\\
    \text{with } \overrightarrow{\gamma} & = & \arg\min_\overrightarrow{\gamma} f\left(x_0 - \sum_{i=1}^{T} \gamma_i d_i\right),
\end{aligned}
```
where $T$ denotes the number of directions $d_i$. This operation can equivalently be described
in terms of the following conditions:

```math
\begin{aligned}
    x - x_0 & \in & \text{span}\left\{d_1,\ldots,d_T\right\}, \\
    \nabla f(x) & \perp & \text{span}\left\{d_1,\ldots,d_T\right\}.
\end{aligned}
```
In this routine, we instead constrain $x_{t}$ and $\nabla f(x_{t})$ to satisfy

```math
\begin{aligned}
    \forall i=1,\ldots,T: & \left< \nabla f(x);\, d_i \right>  & = & 0,\\
    \text{and } & \left< \nabla f(x);\, x - x_0 \right> & = & 0,
\end{aligned}
```
which is a relaxation of the true line/span search conditions.

# Arguments
- `x0`: the starting point.
- `f`: the function on which the (sub)gradient will be evaluated.
- `directions`: the list of all directions required to be orthogonal to the (sub)gradient of x.

# Returns
- `x`: such that all vectors in directions are orthogonal to the (sub)gradient of f at x.
- `gx`: a (sub)gradient of f at x.
- `fx`: the function f evaluated at x.

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
function exact_linesearch_step!(x0::AbstractPoint, f::AbstractFunction, directions)

    x = Point()


    gx, fx = oracle!(f, x)


    add_constraint!(f, (x - x0) * gx == 0)
    for d in directions
        add_constraint!(f, d * gx == 0)
    end


    return x, gx, fx
end
