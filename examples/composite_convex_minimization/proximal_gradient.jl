using PEPit, OrderedCollections, Clarabel


@doc raw"""
    wc_proximal_gradient(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_proximal_gradient`.

Consider the composite convex minimization problem

```math
F_\star \triangleq \min_x \{F(x) \equiv f_1(x) + f_2(x)\},
```

where $f_1$ is $L$-smooth and $\mu$-strongly convex,
and where $f_2$ is closed convex and proper.

# Performance metric

This code computes a worst-case guarantee for the **proximal gradient** method (PGM).
That is, it computes the smallest possible $\tau(n, L, \mu)$ such that the guarantee

```math
\|x_n - x_\star\|^2 \leqslant \tau(n, L, \mu) \|x_0 - x_\star\|^2,
```

is valid, where $x_n$ is the output of the **proximal gradient**,
and where $x_\star$ is a minimizer of $F$.
In short, for given values of $n$, $L$ and $\mu$,
$\tau(n, L, \mu)$ is computed as the worst-case value of
$\|x_n - x_\star\|^2$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm
Proximal gradient is described by

```math
    \begin{aligned}
        y_t & = & x_t - \gamma \nabla f_1(x_t), \\
        x_{t+1} & = & \arg\min_x \left\{f_2(x)+\frac{1}{2\gamma}\|x-y_t\|^2 \right\},
    \end{aligned}
```
for $t \in \{ 0, \dots, n-1\}$ and where $\gamma$ is a step-size.

# Theoretical guarantee
It is well known that a **tight** guarantee for PGM is provided by

```math
\|x_n - x_\star\|^2 \leqslant \max\{(1-L\gamma)^2,(1-\mu\gamma)^2\}^n \|x_0 - x_\star\|^2,
```

which can be found in, e.g., [1, Theorem 3.1]. It is a folk knowledge and the result can be found in many references
for gradient descent; see, e.g.,[2, Section 1.4: Theorem 3], [3, Section 5.1] and [4, Section 4.4].

# References


[[1] A. Taylor, J. Hendrickx, F. Glineur (2018).
Exact worst-case convergence rates of the proximal gradient method for composite convex minimization.
Journal of Optimization Theory and Applications, 178(2), 455-476.](https://arxiv.org/pdf/1705.04398.pdf)

[[2] B. Polyak (1987).
Introduction to Optimization.
Optimization Software New York.](https://www.researchgate.net/profile/Boris-Polyak-2/publication/342978480_Introduction_to_Optimization/links/5f1033e5299bf1e548ba4636/Introduction-to-Optimization.pdf)

[[3] E. Ryu, S. Boyd (2016).
A primer on monotone operator methods.
Applied and Computational Mathematics 15(1), 3-43.](https://web.stanford.edu/~boyd/papers/pdf/monotone_primer.pdf)

[[4] L. Lessard, B. Recht, A. Packard (2016).
Analysis and design of optimization algorithms via integral quadratic constraints.
SIAM Journal on Optimization 26(1), 57-95.](https://arxiv.org/pdf/1408.3595.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value.
- `theoretical_tau`: theoretical value.

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_proximal_gradient(1.0, 0.1, 1.0, 2; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (0.6561, 0.6561)
```
"""
function wc_proximal_gradient(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    f1 = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L))
    f2 = declare_function!(problem, ConvexFunction, OrderedDict())
    func = f1 + f2


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for _ in 1:n

        y = x - gamma * gradient!(f1, x)

        x, _, _ = proximal_step!(y, f2, gamma)
    end


    set_performance_metric!(problem, (x - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = max((1 - mu * gamma)^2, (1 - L * gamma)^2)^n


    if verbose
        println("*** Example file: worst-case performance of the Proximal Gradient Method in function values***")
        println("\tPEPit guarantee:\t ||x_n - x_*||^2 <= $(round(pepit_tau, digits=6)) ||x0 - xs||^2")
        println("\tTheoretical guarantee:\t ||x_n - x_*||^2 <= $(round(theoretical_tau, digits=6)) ||x0 - xs||^2")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_proximal_gradient(1.0, 0.1, 1.0, 2; solver=Clarabel.Optimizer, verbose=true)
