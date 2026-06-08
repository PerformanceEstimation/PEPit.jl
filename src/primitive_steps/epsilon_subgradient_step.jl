@doc raw"""
    epsilon_subgradient_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real)

Create the symbolic primitive step `epsilon_subgradient_step!`.

This routine performs a step $x \leftarrow x_0 - \gamma g_0$
where $g_0 \in\partial_{\varepsilon} f(x_0)$. That is, $g_0$ is an
$\varepsilon$-subgradient of $f$ at $x_0$. The set $\partial_{\varepsilon} f(x_0)$
(referred to as the $\varepsilon$-subdifferential) is defined as (see [1, Section 3])

```math
\partial_{\varepsilon} f(x_0)=\left\{g_0:\,\forall z,\, f(z)\geqslant f(x_0)+\left< g_0;\, z-x_0 \right>-\varepsilon \right\}.
```

An alternative characterization of $g_0 \in\partial_{\varepsilon} f(x_0)$ consists in writing

```math
f(x_0)+f^*(g_0)-\left< g_0;x_0\right>\leqslant \varepsilon.
```

References:
    [[1] A. Brndsted, R.T. Rockafellar.
    On the subdifferentiability of convex functions.
    Proceedings of the American Mathematical Society 16(4), 605-611 (1965)](https://www.jstor.org/stable/2033889)

# Arguments
- `x0`: starting point x0.
- `f`: a function.
- `gamma`: step-size parameter.

# Returns
- `x`: the output point.
- `g0`: an $\varepsilon$-subgradient of f at x0.
- `f0`: the value of the function f at x0.
- `epsilon`: the value of epsilon.

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
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
