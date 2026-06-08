@doc raw"""
    shifted_optimization_step!(dir::AbstractPoint, f::AbstractFunction)

Create the symbolic primitive step `shifted_optimization_step!`.

This routine outputs a stationary point of a minimization problem:

```math
\arg\min_{x} f(x)-\left< \text{dir};\, x \right>.
```

That is, it outputs $x$ such that

```math
\text{dir} \in \partial f(x).
```

Shifted optimization oracles are classically used in difference-of-convex algorithms
(a.k.a., convex-concave procedure), see, e.g., [1].

References:
    [[1] H.A. Le Thi, T. Pham Dinh (2018).
    DC programming and DCA: thirty years of developments.
    Mathematical Programming, 169(1), 5-68.](https://link.springer.com/article/10.1007/s10107-018-1235-y)

# Arguments
- `dir`: direction/linear shift in the objective of the optimization problem
- `f`: function

# Returns
- `x`: oracle output.
- `gx`: the (sub)gradient of f at x.
- `fx`: the function value of f at x.

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
function shifted_optimization_step!(dir::AbstractPoint, f::AbstractFunction)

    x = Point()
    gx = dir
    fx = Expression()


    add_point!(_get_pep_func(f), (x, gx, fx))


    return x, gx, fx
end
