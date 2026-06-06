using PEPit, OrderedCollections, Mosek, MosekTools

function wc_cyclic_coordinate_descent(L, n; solver=Mosek.Optimizer, verbose=true)

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
pepit_tau, theoretical_tau = wc_cyclic_coordinate_descent(L, 9; solver=Mosek.Optimizer, verbose=true)
