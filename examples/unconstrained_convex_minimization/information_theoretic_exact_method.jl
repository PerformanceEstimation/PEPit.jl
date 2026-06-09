using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_information_theoretic(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_information_theoretic`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth and $\mu$-strongly convex ($\mu$ is possibly 0).

# Performance metric

This code computes a worst-case guarantee for the **information theoretic exact method** (ITEM).
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
\|z_n - x_\star\|^2 \leqslant \tau(n, L, \mu) \|z_0 - x_\star\|^2
```

is valid, where $z_n$ is the output of the ITEM,
and where $x_\star$ is the minimizer of $f$.
In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$\|z_n - x_\star\|^2$ when $\|z_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

For $t\in\{0,1,\ldots,n-1\}$, the information theoretic exact method of this example is provided by

```math
    \begin{aligned}
        y_{t} & = & (1-\beta_t) z_t + \beta_t x_t \\
        x_{t+1} & = & y_t - \frac{1}{L} \nabla f(y_t) \\
        z_{t+1} & = & \left(1-q\delta_t\right) z_t+q\delta_t y_t-\frac{\delta_t}{L}\nabla f(y_t),
    \end{aligned}
```
with $y_{-1}=x_0=z_0$, $q=\frac{\mu}{L}$ (inverse condition ratio), and the scalar sequences:

```math
    \begin{aligned}
        A_{t+1} & = & \frac{(1+q)A_t+2\left(1+\sqrt{(1+A_t)(1+qA_t)}\right)}{(1-q)^2},\\
        \beta_{t+1} & = & \frac{A_t}{(1-q)A_{t+1}},\\
        \delta_{t+1} & = & \frac{1}{2}\frac{(1-q)^2A_{t+1}-(1+q)A_t}{1+q+q A_t},
    \end{aligned}
```
with $A_0=0$.

# Theoretical guarantee

A tight worst-case guarantee can be found in [1, Theorem 3]:

```math
\|z_n - x_\star\|^2 \leqslant \frac{1}{1+q A_n} \|z_0-x_\star\|^2,
```

where tightness is obtained on some quadratic loss functions (see [1, Lemma 2]).

# References


[[1] A. Taylor, Y. Drori (2022).
An optimal gradient method for smooth strongly convex minimization.
Mathematical Programming.](https://arxiv.org/pdf/2101.09741.pdf)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_information_theoretic(0.001, 1.0, 15; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.756605, 0.756605)
```
"""
function wc_information_theoretic(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)


    z0 = set_initial_point!(problem)


    set_initial_condition!(problem, (z0 - xs)^2 <= 1)


    A_new = 0.0
    q = mu / L

    x = z0
    z = z0

    for i in 1:n
        A_old = A_new
        A_new = ((1 + q) * A_old + 2 * (1 + sqrt((1 + A_old) * (1 + q * A_old)))) / (1 - q)^2
        beta = A_old / (1 - q) / A_new
        delta = 1 / 2 * ((1 - q)^2 * A_new - (1 + q) * A_old) / (1 + q + q * A_old)

        y = (1 - beta) * z + beta * x
        x = y - 1 / L * gradient!(func, y)
        z = (1 - q * delta) * z + q * delta * y - delta / L * gradient!(func, y)
    end


    set_performance_metric!(problem, (z - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = 1 / (1 + q * A_new)


    if verbose
        println("*** Example file: worst-case performance of the information theoretic exact method ***")
        println("\tPEP-it guarantee:\t ||z_n - x_* ||^2 <= $(round(pepit_tau, digits=6)) ||z_0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||z_n - x_* ||^2 <= $(round(theoretical_tau, digits=6)) ||z_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_information_theoretic(0.001, 1.0, 15; solver=Clarabel.Optimizer, verbose=true)
