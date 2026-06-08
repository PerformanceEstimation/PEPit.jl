@doc raw"""
    inexact_gradient_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real, epsilon::Real; notion::String="absolute")

Create the symbolic primitive step `inexact_gradient_step!`.

This routine performs a step $x \leftarrow x_0 - \gamma d_{x_0}$
where $d_{x_0}$ is close to the gradient of $f$ in $x_0$
in the following sense:

```math
\|d_{x_0} - \nabla f(x_0)\|^2 \leqslant \left\{
          \begin{aligned}
              & \varepsilon^2                        & \text{if notion is set to 'absolute'}, \\
              & \varepsilon^2 \|\nabla f(x_0)\|^2 & \text{if notion is set to 'relative'}.
          \end{aligned}
          \right.
```

This relative approximation is used at least in 3 PEPit examples,
in particular in 2 unconstrained convex minimizations:
an inexact gradient descent, and an inexact accelerated gradient.

References:
    [[1] E. De Klerk, F. Glineur, A. Taylor (2020).
    Worst-case convergence analysis of inexact gradient and Newton methods
    through semidefinite programming performance estimation.
    SIAM Journal on Optimization, 30(3), 2053-2082.](https://arxiv.org/pdf/1709.05191.pdf)

# Arguments
- `x0`: starting point x0.
- `f`: a function.
- `gamma`: step-size parameter.
- `epsilon`: the required accuracy.
- `notion`: defines the mode (absolute or relative inaccuracy). By default, `notion="absolute"`.

# Returns
- `x`: the output point.
- `dx0`: the approximate (sub)gradient of f at x0.
- `fx0`: the value of the function f at x0.

# Throws
- `ErrorException` (via `error`): if `notion` is not one of `"absolute"` or `"relative"`.

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
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
