using PEPit, OrderedCollections, Clarabel


@doc raw"""
    wc_accelerated_proximal_gradient(mu, L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_accelerated_proximal_gradient`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f(x) + h(x)\},
```

where $f$ is $L$-smooth and $\mu$-strongly convex,
and where $h$ is closed convex and proper.

# Performance metric

This code computes a worst-case guarantee for the **accelerated proximal gradient** method,
also known as **fast proximal gradient (FPGM)** method or FISTA [1].
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
F(x_n) - F(x_\star) \leqslant \tau(n, L, \mu) \|x_0 - x_\star\|^2,
```

is valid, where $x_n$ is the output of the **accelerated proximal gradient** method,
and where $x_\star$ is a minimizer of $F$.

In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$F(x_n) - F(x_\star)$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
Initialize $\lambda_1=1$, $y_1=x_0$. One iteration of FISTA is described by

```math
\begin{aligned}
    \text{Set: }\lambda_{t+1} & = & \frac{1 + \sqrt{4\lambda_t^2 + 1}}{2}\\
    x_t & = & \arg\min_x \left\{h(x)+\frac{L}{2}\|x-\left(y_t - \frac{1}{L} \nabla f(y_t)\right)\|^2 \right\}\\
    y_{t+1} & = & x_t + \frac{\lambda_t-1}{\lambda_{t+1}} (x_t-x_{t-1}).
\end{aligned}
```
# Theoretical guarantee
The following worst-case guarantee can be found in e.g., [1, Theorem 4.4]:

```math
f(x_n)-f_\star \leqslant \frac{L}{2}\frac{\|x_0-x_\star\|^2}{\lambda_n^2}.
```

# References


[[1] A. Beck, M. Teboulle (2009).
A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
SIAM journal on imaging sciences, 2009, vol. 2, no 1, p. 183-202.](https://www.ceremade.dauphine.fr/~carlier/FISTA)

# Arguments
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_accelerated_proximal_gradient(0.0, 1.0, 4; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_accelerated_proximal_gradient(mu, L, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    h = declare_function!(problem, ConvexFunction, OrderedDict())
    F = f + h


    xs = stationary_point!(F)
    Fs = value!(F, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x_new = x0
    y = x0
    lam = 1
    lam_old = lam
    local hx_new::Expression
    for i in 0:(n-1)

        lam_old = lam
        lam = (1 + sqrt(4 * lam_old^2 + 1)) / 2


        x_old = x_new


        x_new, _, hx_new = proximal_step!(y - 1 / L * gradient!(f, y), h, 1 / L)


        y = x_new + (lam_old - 1) / lam * (x_new - x_old)
    end


    set_performance_metric!(problem, (value!(f, x_new) + hx_new) - Fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = L / (2 * lam_old^2)

    if mu != 0
        println("Warning: momentum is tuned for non-strongly convex functions.")
    end


    if verbose
        println("*** Example file: worst-case performance of the Accelerated Proximal Gradient Method in function values***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t f(x_n)-f_* <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_accelerated_proximal_gradient(0.0, 1.0, 4; solver=Clarabel.Optimizer, verbose=true)
