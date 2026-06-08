using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_accelerated_proximal_point(A0, gammas, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_proximal_point`.

Consider the minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is  convex and possibly non-smooth.

# Performance metric

This code computes a worst-case guarantee an **accelerated proximal point** method,
aka **fast proximal point** method (FPP).
That is, it computes the smallest possible $\tau(n, A_0,\vec{\gamma})$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, A_0, \vec{\gamma}) \left(f(x_0) - f_\star + \frac{A_0}{2}  \|x_0 - x_\star\|^2\right)
```

is valid, where $x_n$ is the output of FPP (with step-size $\gamma_t$ at step
$t\in \{0, \dots, n-1\}$) and where $x_\star$ is a minimizer of $f$
and $A_0$ is a positive number.

In short, for given values of $n$,  $A_0$ and $\vec{\gamma}$, $\tau(n)$
is computed as the worst-case value of $f(x_n)-f_\star$
when $f(x_0) - f_\star + \frac{A_0}{2} \|x_0 - x_\star\|^2 \leqslant 1$, for the following method.

# Algorithm

For $t\in \{0, \dots, n-1\}$:

```math
   \begin{aligned}
       y_{t+1} & = & (1-\alpha_{t} ) x_{t} + \alpha_{t} v_t \\
       x_{t+1} & = & \arg\min_x \left\{f(x)+\frac{1}{2\gamma_t}\|x-y_{t+1}\|^2 \right\}, \\
       v_{t+1} & = & v_t + \frac{1}{\alpha_{t}} (x_{t+1}-y_{t+1})
   \end{aligned}
```
with

```math
   \begin{aligned}
       \alpha_{t} & = & \frac{\sqrt{(A_t \gamma_t)^2 + 4 A_t \gamma_t} - A_t \gamma_t}{2} \\
       A_{t+1} & = & (1 - \alpha_{t}) A_t
   \end{aligned}
```
and $v_0=x_0$.



# Theoretical guarantee

A theoretical **upper** bound can be found in [1, Theorem 2.3.]:

```math
f(x_n)-f_\star \leqslant \frac{4}{A_0 (\sum_{t=0}^{n-1} \sqrt{\gamma_t})^2}\left(f(x_0) - f_\star + \frac{A_0}{2}  \|x_0 - x_\star\|^2 \right).
```

# References

The accelerated proximal point was first obtained and analyzed in [1].

[[1] O. Guler (1992).
New proximal point algorithms for convex minimization.
SIAM Journal on Optimization, 2(4):649-664.](https://epubs.siam.org/doi/abs/10.1137/0802032?mobileUi=0)

# Arguments
- `A0`: initial value for parameter A_0.
- `gammas`: sequence of step-sizes.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_accelerated_proximal_point(5, [(i + 1) / 1.1 for i in 0:2], 3; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_accelerated_proximal_point(A0, gammas, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    func = declare_function!(problem, ConvexFunction, OrderedDict())


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, value!(func, x0) - fs + A0 / 2 * (x0 - xs)^2 <= 1)


    x, v = x0, x0
    A = A0
    for i in 1:n
        alpha = (sqrt((A * gammas[i])^2 + 4 * A * gammas[i]) - A * gammas[i]) / 2
        y = (1 - alpha) * x + alpha * v
        x, _, _ = proximal_step!(y, func, gammas[i])
        v = v + 1 / alpha * (x - y)
        A = (1 - alpha) * A
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    accumulation = 0.0
    for i in 1:n
        accumulation += sqrt(gammas[i])
    end
    theoretical_tau = 4 / A0 / accumulation^2


    if verbose
        println("*** Example file: worst-case performance of fast proximal point method ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) (f(x_0) - f_* + A/2* ||x_0 - x_*||^2)")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_proximal_point(5, [(i + 1) / 1.1 for i in 0:2], 3; solver=Clarabel.Optimizer, verbose=true)
