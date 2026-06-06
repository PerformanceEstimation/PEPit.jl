using PEPit, OrderedCollections, Clarabel

function wc_saga(L, mu, n; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    h = declare_function!(problem, ConvexFunction, OrderedDict())
    fn = [declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu); reuse_gradient=true) for _ in 1:n]


    func = h + sum(fn) / n


    xs = stationary_point!(func)


    phi = [set_initial_point!(problem) for _ in 1:n]
    x0 = set_initial_point!(problem)


    gamma = 1 / 2 / (mu * n + L)
    c = 1 / 2 / gamma / (1 - mu * gamma) / n
    g, f = Vector{Any}(undef, n), Vector{Any}(undef, n)
    g0, f0 = Vector{Any}(undef, n), Vector{Any}(undef, n)
    gs, fs = Vector{Any}(undef, n), Vector{Any}(undef, n)
    init_lyapunov = c * (xs - x0)^2

    for i in 1:n
        g[i], f[i] = oracle!(fn[i], phi[i])
        gs[i], fs[i] = oracle!(fn[i], xs)
        init_lyapunov = init_lyapunov + 1 / n * (f[i] - fs[i] - gs[i] * (phi[i] - xs))
    end


    set_initial_condition!(problem, init_lyapunov <= 1)


    final_lyapunov_avg = (xs - xs)^2
    for i in 1:n
        g0[i], f0[i] = oracle!(fn[i], x0)
        w = x0 - gamma * (g0[i] - g[i])
        for j in 1:n
            w = w - gamma / n * g[j]
        end
        x1, _, _ = proximal_step!(w, h, gamma)
        final_lyapunov = c * (x1 - xs)^2
        for j in 1:n
            if i != j
                gi, fi = g[j], f[j]
                final_lyapunov = final_lyapunov + 1 / n * (fi - fs[j] - gs[j] * (phi[j] - xs))
            else
                gi, fi = g0[i], f0[i]
                final_lyapunov = final_lyapunov + 1 / n * (fi - fs[j] - gs[j] * (x0 - xs))
            end
        end
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n
    end


    set_performance_metric!(problem, final_lyapunov_avg)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    theoretical_tau = (1 - gamma * mu)


    if verbose
        println("*** Example file: worst-case performance of SAGA for Lyapunov function V_t ***")
        println("\tPEPit guarantee:\t V^(1) <= $(round(pepit_tau, digits=6)) V^(0)")
        println("\tTheoretical guarantee:\t V^(1) <= $(round(theoretical_tau, digits=6)) V^(0)")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_saga(1.0, 0.1, 5; solver=Clarabel.Optimizer, verbose=true)
