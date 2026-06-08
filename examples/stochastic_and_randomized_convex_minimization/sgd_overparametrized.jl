using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_sgd_overparametrized(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_sgd_overparametrized`.

Consider the finite sum minimization problem

```math
F_\star \triangleq \min_x \left\{F(x) \equiv \frac{1}{n} \sum_{i=1}^n f_i(x)\right\},
```

where $f_1, ..., f_n$ are $L$-smooth and $\mu$-strongly convex. In the sequel, we use the notation
$\mathbb{E}$ for denoting the expectation over the uniform distribution of the index
$i \sim \mathcal{U}\left([|1, n|]\right)$, e.g., $F(x)\equiv\mathbb{E}[f_i(x)]$.
In addition, we assume a zero variance at the optimal point (which is denoted by $x_\star$):

```math
\mathbb{E}\left[\|\nabla f_i(x_\star)\|^2\right] = \frac{1}{n} \sum_{i=1}^n \|\nabla f_i(x_\star)\|^2 = 0,
```

where the expectation $\mathbb{E}$ is taken over the uniform distribution of the index
$i \sim \mathcal{U}\left([|1, n|]\right)$.

This kind of situations happens for example in machine learning in the interpolation regime,
that is if there exists a model $x_\star$
such that the loss $\mathcal{L}$ on any observation $(z_i)_{i \in [|1, n|]}$,
$\mathcal{L}(x_\star, z_i) = f_i(x_\star)$ is zero.

# Performance metric

This code computes a worst-case guarantee for one step of the **stochastic gradient descent** (SGD) in expectation,
for the distance to optimal point.
That is, it computes the smallest possible $\tau(L, \mu, \gamma, n)$ such that

```math
\mathbb{E}\left[\|x_1 - x_\star\|^2\right] \leqslant \tau(L, \mu, \gamma, n) \|x_0 - x_\star\|^2
```

is valid, where $x_1$ is the output of one step of SGD.

# Algorithm
One iteration of SGD is described by:

```math
\begin{aligned}
    \text{Pick random }i & \sim & \mathcal{U}\left([|1, n|]\right), \\
    x_{t+1} & = & x_t - \gamma \nabla f_{i}(x_t),
\end{aligned}
```
where $\gamma$ is a step-size.

# Theoretical guarantee
An empirically tight one-iteration guarantee is provided in the code of PESTO [1]:

```math
\mathbb{E}\left[\|x_1 - x_\star\|^2\right] \leqslant \left(\max(1 - \gamma\mu, L\gamma - 1\right)\right)^2 \|x_0-x_\star\|^2.
```

Note that we observe the guarantee does not depend on the number $n$ of
functions for this particular setting, thereby implying that the guarantees are also valid for expectation
minimization settings (i.e., when $n$ goes to infinity).

# References
Empirically tight guarantee provided in code of [1]. Using SDPs for analyzing SGD-type method was
proposed in [2, 3].

[[1] A. Taylor, J. Hendrickx, F. Glineur (2017). Performance Estimation Toolbox (PESTO): automated worst-case
analysis of first-order optimization methods. In 56th IEEE Conference on Decision and Control (CDC).](https://github.com/AdrienTaylor/Performance-Estimation-Toolbox)

[[2] B. Hu, P. Seiler, L. Lessard (2020). Analysis of biased stochastic gradient descent using sequential
semidefinite programs. Mathematical programming.](https://arxiv.org/pdf/1711.00987.pdf)

[[3] A. Taylor, F. Bach (2019). Stochastic first-order methods: non-asymptotic and computer-aided analyses
via potential functions. Conference on Learning Theory (COLT).](https://arxiv.org/pdf/1902.00947.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `mu`: strong convexity or monotonicity parameter, as used by the modeled class.
- `gamma`: step-size parameter.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: theoretical value

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_sgd_overparametrized(1.0, 0.1, 2.3, 5; solver=Clarabel.Optimizer, verbose=true)
```
"""
function wc_sgd_overparametrized(L, mu, gamma, n; solver=Clarabel.Optimizer, verbose=true)
    problem = PEP()


    fn = [declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu)) for _ in 1:n]
    func = sum(fn) / n


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    var = sum(gradient!(f, xs)^2 for f in fn) / n
    add_constraint!(problem, var <= 0.0)
    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    distavg = sum((x0 - gamma * gradient!(f, x0) - xs)^2 for f in fn) / n

    set_performance_metric!(problem, distavg)

    pepit_tau = solve!(problem; solver=solver, verbose=verbose)

    theoretical_tau = max(1 - gamma * mu, L * gamma - 1)^2

    if verbose
        println("*** Example file: worst-case performance of stochastic gradient descent with fixed step-size and with zero variance at the optimal point ***")
        println("\tPEPit guarantee:\t E[||x_1 - x_*||^2] <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t E[||x_1 - x_*||^2] <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_sgd_overparametrized(1.0, 0.1, 2.3, 5; solver=Clarabel.Optimizer, verbose=true)
