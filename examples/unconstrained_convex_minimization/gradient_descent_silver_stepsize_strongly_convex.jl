using PEPit, OrderedCollections, Mosek, MosekTools

function wc_gradient_descent_silver_stepsize_strongly_convex(L, mu, n; solver=Mosek.Optimizer, verbose=true)


    if !isinteger(log2(n))
        @warn "Silver step-size strategy is optimally designed when n is a power of 2." *
              " The provided input n is not a power of 2." *
              " We decompose n as sum_k 2^k and recursely use sequences of stepsizes of length 2^k."
    end


    nbits = ndigits(n, base=2)
    n_glue_list = [i for i in 0:(nbits - 1) if (n & (1 << i)) != 0]


    h = Float64[]
    theoretical_tau = 1.0


    psi(t) = (1 + L / mu * t) / (1 + t)


    for n_glue in n_glue_list


        y = [mu / L]
        z = [mu / L]

        a = [psi(y[1])]
        b = [psi(z[1])]

        h_temp = [b[1]]
        for step in 1:n_glue
            z_old = z[step]
            eta = 1 - z_old
            y_new = z_old / (eta + sqrt(1 + eta^2))
            z_new = z_old * (eta + sqrt(1 + eta^2))
            push!(y, y_new)
            push!(z, z_new)
            a_new = psi(y_new)
            b_new = psi(z_new)
            push!(a, a_new)
            push!(b, b_new)
            h_tilde = h_temp[1:end-1]
            h_temp = vcat(h_tilde, [a_new], h_tilde, [b_new])
        end


        h = vcat(h, h_temp)


        theoretical_tau *= ((1 - z[end]) / (1 + z[end]))^2
    end


    problem = PEP()


    param = OrderedDict("mu" => mu, "L" => L)
    func = declare_function!(problem, SmoothStronglyConvexFunction, param)


    xs = stationary_point!(func)


    x0 = set_initial_point!(problem)


    set_initial_condition!(problem, (x0 - xs)^2 <= 1)


    x = x0
    for i in 1:n
        x = x - h[i] / L * gradient!(func, x)
    end


    set_performance_metric!(problem, (x - xs)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    if verbose
        println("*** Example file: worst-case performance of gradient descent with silver step-sizes ***")
        println("\tPEPit guarantee:\t ||x_n - x_*||^2 <= $(round(pepit_tau, digits=6)) ||x_0 - x_*||^2")
        println("\tTheoretical guarantee:\t ||x_n - x_*||^2 <= $(round(theoretical_tau, digits=6)) ||x_0 - x_*||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau = wc_gradient_descent_silver_stepsize_strongly_convex(3.2, 0.1, 8; verbose=true)
