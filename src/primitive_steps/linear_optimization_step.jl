@doc raw"""
    linear_optimization_step!(dir::AbstractPoint, ind::AbstractFunction)

Create the symbolic primitive step `linear_optimization_step!`.

This routine outputs the result of a minimization problem with linear objective (whose direction
is provided by `dir`) on the domain of the (closed convex) indicator function `ind`.
That is, it outputs a solution to

```math
\arg\min_{\text{ind}(x)=0} \left< \text{dir};\, x \right>,
```

One can notice that $x$ is solution of this problem if and only if

```math
- \text{dir} \in \partial \text{ind}(x).
```

Linear optimization oracles are classically used in conditional gradient-type algorithm (a.k.a., Frank-Wolfe) [1].

References:
    [[1] M. Frank, P. Wolfe (1956).
    An algorithm for quadratic programming.
    Naval research logistics quarterly, 3(1-2), 95-110.](https://arxiv.org/pdf/1608.04826.pdf)

# Arguments
- `dir`: direction of optimization
- `ind`: convex indicator function

# Returns
- `x`: oracle output.
- `gx`: the (sub)gradient of ind on x.
- `fx`: the function value of ind on x.

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
function linear_optimization_step!(dir::AbstractPoint, ind::AbstractFunction)

    x = Point()
    gx = -dir
    fx = Expression()


    add_point!(_get_pep_func(ind), (x, gx, fx))


    return x, gx, fx
end
