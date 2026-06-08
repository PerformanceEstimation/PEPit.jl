using PEPit, OrderedCollections


@doc raw"""
    wc_accelerated_inexact_forward_backward(L, zeta, n; verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_inexact_forward_backward`.

Consider the composite convex minimization problem,

```math
F_\star \triangleq \min_x \left\{F(x) \equiv f(x) + g(x) \right\},
```

where $f$ is $L$-smooth convex, and $g$ is closed, proper, and convex.
We further assume that one can readily evaluate the gradient of $f$ and that one has access to an inexact
version of the proximal operator of $g$ (whose level of accuracy is controlled by some
parameter $\zeta\in (0,1)$).

# Performance metric

This code computes a worst-case guarantee for an **accelerated inexact forward backward** (AIFB) method (a.k.a.,
inexact accelerated proximal gradient method). That is, it computes the smallest possible
$\tau(n, L, \zeta)$ such that the guarantee

```math
F(x_n) - F(x_\star) \leqslant \tau(n, L, \zeta) \|x_0 - x_\star\|^2,
```

is valid, where $x_n$ is the output of the IAFB, and where $x_\star$ is a minimizer of $F$.

In short, for given values of $n$, $L$ and $\zeta$, $\tau(n, L, \zeta)$ is computed as
the worst-case value of $F(x_n) - F(x_\star)$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
Let $t\in\{0,1,\ldots,n\}$. The method is presented in, e.g., [1, Algorithm 3.1].
For simplicity, we instantiate [1, Algorithm 3.1] using simple values for its parameters and for the problem
setting (in the notation of [1]: $A_0\triangleq 0$, $\mu=0$, $\xi_t \triangleq0$,
$\sigma_t\triangleq 0$, $\lambda_t \triangleq\gamma\triangleq\tfrac{1}{L}$,
$\zeta_t\triangleq\zeta$, $\eta \triangleq (1-\zeta^2) \gamma$), and without backtracking,
arriving to:

```math
    \begin{aligned}
         A_{t+1} && = A_t + \frac{\eta+\sqrt{\eta^2+4\eta A_t}}{2},\\
         y_{t} && = x_t + \frac{A_{t+1}-A_t}{A_{t+1}} (z_t-x_t),\\
         (x_{t+1},v_{t+1}) && \approx_{\varepsilon_t} \left(\mathrm{prox}_{\gamma g}\left(y_t-\gamma \nabla f(y_t)\right),\,
         \mathrm{prox}_{ g^*/\gamma}\left(\frac{y_t-\gamma \nabla f(y_t)}{\gamma}\right)\right),\\
         && \text{with } \varepsilon_t = \frac{\zeta^2\gamma^2}{2}\|v_{t+1}+\nabla f(y_t) \|^2,\\
         z_{t+1} && = z_t-(A_{t+1}-A_t)\left(v_{t+1}+\nabla f(y_t)\right),\\
    \end{aligned}
```
where $\{\varepsilon_t\}_{t\geqslant 0}$ is some sequence of accuracy parameters (whose values are fixed
within the algorithm as it runs), and $\{A_t\}_{t\geqslant 0}$ is some scalar sequence of parameters
for the method (typical of accelerated methods).

The line with "$\approx_{\varepsilon}$" can be described as the pair $(x_{t+1},v_{t+1})$ satisfying
an accuracy requirement provided by [1, Definition 2.3]. More precisely (but without providing any intuition),
it requires the existence of some $w_{t+1}$ such that $v_{t+1} \in \partial g(w_{t+1})$
and for which the accuracy requirement

```math
\gamma^2 || x_{t+1} - y_t + \gamma v_{t+1} ||^2 + \gamma  (g(x_{t+1}) - g(w_{t+1}) - v_{t+1}(x_{t+1} - w_{t+1})) \leqslant \varepsilon_t,
```

is valid.

# Theoretical guarantee
A theoretical upper bound is obtained in [1, Corollary 3.5]:

```math
F(x_n)-F_\star\leqslant \frac{2L \|x_0-x_\star\|^2}{(1-\zeta^2)n^2}.
```

# References
The method and theoretical result can be found in [1, Section 3].

[[1] M. Barre, A. Taylor, F. Bach (2021).
A note on approximate accelerated forward-backward methods with
absolute and relative errors, and possibly strongly convex objectives.
arXiv:2106.15536v2.](https://arxiv.org/pdf/2106.15536v2.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `zeta`: relative approximation parameter in (0,1).
- `n`: number of iterations.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_accelerated_inexact_forward_backward(1.3, 0.45, 11; verbose=true)
```
"""
function wc_accelerated_inexact_forward_backward(L, zeta, n; verbose=true)


    problem = PEP()


    f = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => L); reuse_gradient=true)
    h = declare_function!(problem, ConvexFunction, OrderedDict(); reuse_gradient=false)


    F = f + h


    xs = stationary_point!(F)


    Fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    gamma = 1 / L


    eta = (1 - zeta^2) * gamma


    A = 0.0


    x = x0
    z = x0


    hx = nothing


    for _ in 1:n

        A_next = A + (eta + sqrt(eta^2 + 4 * eta * A)) / 2


        y = x + (1 - A / A_next) * (z - x)


        gy = gradient!(f, y)


        x, _, hx, _, vx, _, eps_var = inexact_proximal_step!(y - gamma * gy, h, gamma; opt="PD_gapI")


        add_constraint!(h, eps_var <= (zeta * gamma)^2 / 2 * (vx + gy)^2)


        z = z - (A_next - A) * (vx + gy)


        A = A_next
    end


    set_performance_metric!(problem, value!(f, x) + hx - Fs)


    pepit_tau = solve!(problem; verbose=verbose)


    theoretical_tau = 2 * L / (1 - zeta^2) / n^2


    if verbose

        println("*** Example file: worst-case performance of an inexact accelerated forward backward method ***")
        println("\tPEPit guarantee:\t F(x_n)-F_* <= $(round(pepit_tau, digits=7)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t F(x_n)-F_* <= $(round(theoretical_tau, digits=7)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_inexact_forward_backward(1.3, 0.45, 11; verbose=true)
