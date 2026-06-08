@doc raw"""
    inexact_proximal_step!(x0::AbstractPoint, f::AbstractFunction, gamma::Real; opt::String="PD_gapII")

Create the symbolic primitive step `inexact_proximal_step!`.

This routine encodes an inexact proximal operation with step size $\gamma$. That is, it outputs a tuple
$(x, g\in \partial f(x), f(x), w, v\in\partial f(w), f(w), \varepsilon)$ which are described as follows.

First, $x$ is an approximation to the proximal point of $x_0$ on function $f$:

```math
x \approx \mathrm{prox}_{\gamma f}(x_0)\triangleq\arg\min_x \left\{ \gamma f(x) + \frac{1}{2}\|x-x_0\|^2\right\},
```

where the meaning of $\approx$ depends on the option "opt" and is explained below. The notions of inaccuracy
implemented within this routine are specified using primal and dual proximal problems, denoted by

```math
    \begin{aligned}
    &\Phi^{(p)}_{\gamma f}(x; x_0) \triangleq \gamma f(x) + \frac{1}{2}\|x-x_0\|^2,\\
    &\Phi^{(d)}_{\gamma f}(v; x_0) \triangleq -\gamma f^*(v)-\frac{1}{2}\|x_0-\gamma v\|^2 + \frac{1}{2}\|x_0\|^2,\\
    \end{aligned}
```
where $\Phi^{(p)}_{\gamma f}(x;x_0)$ and $\Phi^{(d)}_{\gamma f}(v;x_0)$ respectively denote the primal
and the dual proximal problems, and where $f^*$ is the Fenchel conjugate of $f$. The options below
encode different meanings of "$\approx$" by specifying accuracy requirements on primal-dual pairs:

```math
(x,v) \approx_{\varepsilon} \left(\mathrm{prox}_{\gamma f}(x_0),\,\mathrm{prox}_{f^*/\gamma}(x_0/\gamma)\right),
```

where $\approx_{\varepsilon}$ corresponds to require the primal-dual pair $(x,v)$ to satisfy some
primal-dual accuracy requirement:

```math
\Phi^{(p)}_{\gamma f}(x;x_0)-\Phi^{(d)}_{\gamma f}(v;x_0) \leqslant \varepsilon,
```

where $\varepsilon\geqslant 0$ is the error magnitude, which is returned to the user so that one can
constrain it to be bounded by some other values.

**Relation to the exact proximal operation:**  In the exact case (no error in the computation,
$\varepsilon=0$), $v$ corresponds to the solution of the dual proximal problem and one can write

```math
x = x_0-\gamma g,
```

with $g=v=\mathrm{prox}_{f^*/\gamma}(x_0/\gamma)\in\partial f(x)$, and $x=w$.

**Reformulation of the primal-dual gap:** In regard with the exact proximal computation; the inexact case under
consideration here can be described as performing

```math
x = x_0-\gamma v + e,
```

where $v$ is an $\epsilon$-subgradient of $f$ at $x$
(notation $v\in\partial_{\epsilon} f(x)$)
and $e$ is some additional computation error. Those elements allow for a common convenient reformulation of
the primal-dual gap, written in terms of the magnitudes of $\epsilon$ and of $e$:

```math
\Phi^{(p)}_{\gamma f}(x;x_0)-\Phi^{(d)}_{\gamma f}(v;x_0) = \frac{1}{2} \|e\|^2 + \gamma \epsilon.
```

**Options:** The following options are available (a list of such choices is presented in [4]; we provide a reference
for each of those choices below).

    - 'PD_gapI' : the constraint imposed on the output is the vanilla (see, e.g., [2])

```math
\Phi^{(p)}_{\gamma f}(x;x_0)-\Phi^{(d)}_{\gamma f}(v;x_0) \leqslant \varepsilon.
```

    This approximation requirement is used in one PEPit example: an accelerated inexact forward backward.

    - 'PD_gapII' : the constraint is stronger than the vanilla primal-dual gap, as more structure is imposed
                   (see, e.g., [1,5]) :

```math
\Phi^{(p)}_{\gamma f}(x;x_0)-\Phi^{(d)}_{\gamma f}(g;x_0) \leqslant \varepsilon,
```

    where we imposed that $v\triangleq g\in\partial f(x)$ and $w\triangleq x$. This approximation
    requirement is used in two PEPit examples: in a relatively inexact proximal point algorithm and in a partially
    inexact Douglas-Rachford splitting.

    - 'PD_gapIII' : the constraint is stronger than the vanilla primal-dual gap, as more structure is imposed
                    (see, e.g., [3]):

```math
\Phi^{(p)}_{\gamma f}(x;x_0)-\Phi^{(d)}_{\gamma f}(\tfrac{x_0 - x}{\gamma};x_0) \leqslant \varepsilon,
```

    where we imposed that $v \triangleq \frac{x_0 - x}{\gamma}$.

References:

    [[1] R.T. Rockafellar (1976).
    Monotone operators and the proximal point algorithm. SIAM journal on control and optimization, 14(5), 877-898.](https://epubs.siam.org/doi/pdf/10.1137/0314056)

    [[2] R.D. Monteiro, B.F. Svaiter (2013).
    An accelerated hybrid proximal extragradient method for convex optimization
    and its implications to second-order methods.
    SIAM Journal on Optimization, 23(2), 1092-1125.](https://epubs.siam.org/doi/abs/10.1137/110833786)

    [[3] S. Salzo, S. Villa (2012).
    Inexact and accelerated proximal point algorithms.
    Journal of Convex analysis, 19(4), 1167-1192.](http://www.optimization-online.org/DB_FILE/2011/08/3128.pdf)

    [[4] M. Barre, A. Taylor, F. Bach (2020).
    Principled analyses and design of first-order methods with inexact proximal operators.](https://arxiv.org/pdf/2006.06041v3.pdf)

    [[5] A. d'Aspremont, D. Scieur, A. Taylor (2021).
    Acceleration Methods.
    Foundations and Trends in Optimization: Vol. 5, No. 1-2.](https://arxiv.org/pdf/2101.09545.pdf)

# Arguments
- `x0`: point for which we aim to compute an approximate proximal step.
- `f`: function whose proximal operator is approximated.
- `gamma`: step-size parameter.
- `opt`: option (type of error requirement) among 'PD_gapI', 'PD_gapII', and  'PD_gapIII'.

# Returns
- `x`: the approximated proximal point.
- `gx`: a (sub)gradient of f at x (subgradient used in evaluating the accuracy criterion).
- `fx`: f evaluated at x.
- `w`: a point w such that v (see next output) is a subgradient of f at w.
- `v`: the approximated proximal point of the dual problem, (sub)gradient of f evaluated at w.
- `fw`: f evaluated at w.
- `eps_var`: value of the primal-dual gap (which can be further bounded by the user).

See also [`Point`](@ref), [`Expression`](@ref), and [`add_constraint!`](@ref).
"""
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
