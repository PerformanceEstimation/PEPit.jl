using PEPit, OrderedCollections, Clarabel

@doc raw"""
    wc_cyclic_coordinate_descent(L, n; solver=Clarabel.Optimizer, verbose=true)

# Problem statement

Compute a PEPit worst-case guarantee for `wc_cyclic_coordinate_descent`.

Consider the convex minimization problem

```math
f_\star \triangleq \min_x f(x),
```

where $f$ is $L$-smooth by blocks (with $d$ blocks) and convex.

# Performance metric

This code computes a worst-case guarantee for **cyclic coordinate descent** with fixed step-sizes $1/L_i$.
That is, it computes the smallest possible $\tau(n, d, L)$ such that the guarantee

```math
f(x_n) - f_\star \leqslant \tau(n, d, L) \|x_0 - x_\star\|^2
```

is valid, where $x_n$ is the output of cyclic coordinate descent with fixed step-sizes $1/L_i$, and
where $x_\star$ is a minimizer of $f$.

In short, for given values of $n$, $L$, and $d$, $\tau(n, d, L)$ is computed as
the worst-case value of $f(x_n)-f_\star$ when $\|x_0 - x_\star\|^2 \leqslant 1$.

# Algorithm

Cyclic coordinate descent is described by

```math
x_{t+1} = x_t - \frac{1}{L_{i_t}} \nabla_{i_t} f(x_t),
```

where $L_{i_t}$ is the Lipschitz constant of the block $i_t$,
and where $i_t$ follows a prescribed ordering.

# References


[[1] Z. Shi, R. Liu (2016).
Better worst-case complexity analysis of the block coordinate descent method for large scale machine learning.
In 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA).](https://arxiv.org/pdf/1608.04826.pdf)

# Arguments
- `L`: smoothness or Lipschitz parameter, as used by the modeled class.
- `n`: number of iterations.
- `solver`: JuMP optimizer constructor used to solve the generated SDP.
- `verbose`: print example and solver progress information when true.

# Returns
- `pepit_tau`: worst-case value
- `theoretical_tau`: None

# Julia usage
```julia
pepit_tau, theoretical_tau = wc_cyclic_coordinate_descent(L, 9; solver=Clarabel.Optimizer, verbose=true)
# Returns approximately: (pepit_tau, theoretical_tau) = (1.489276, nothing)
```
"""
function wc_cyclic_coordinate_descent(L, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    d = length(L)
    partition = declare_block_partition!(problem, d)


    param = OrderedDict("partition" => partition, "L" => L)
    func = declare_function!(problem, BlockSmoothConvexFunctionCheap, param)


    xs = stationary_point!(func)
    fs = value!(func, xs)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for k in 0:(n - 1)
        i = k % d
        x = x - 1 / L[i + 1] * get_block(partition, gradient!(func, x), i + 1)
    end


    set_performance_metric!(problem, value!(func, x) - fs)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = nothing

    if verbose
        println("*** Example file: worst-case performance of cyclic coordinate descent with fixed step-sizes ***")
        println("\tPEPit guarantee:\t f(x_n)-f_* <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
    end

    return pepit_tau, theoretical_tau
end


L = [1.0, 2.0, 10.0]
pepit_tau, theoretical_tau = wc_cyclic_coordinate_descent(L, 9; solver=Clarabel.Optimizer, verbose=true)
