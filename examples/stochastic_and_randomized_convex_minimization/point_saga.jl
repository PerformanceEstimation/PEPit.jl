using PEPit, OrderedCollections, Clarabel

function wc_point_saga(L, mu, n; solver=Clarabel.Optimizer, verbose=true)

    problem = PEP()


    fn = [declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("L" => L, "mu" => mu)) for _ in 1:n]
    func = sum(fn) / n


    xs = stationary_point!(func)


    phi = [set_initial_point!(problem) for _ in 1:n]
    x0 = set_initial_point!(problem)


    gamma = sqrt((n - 1)^2 + 4 * n * L / mu) / 2 / L / n - (1 - 1 / n) / 2 / L
    c = 1 / (mu * L)


    init_lyapunov = (xs - x0)^2
    gs = [gradient!(fn[i], xs) for i in 1:n]
    for i in 1:n
        init_lyapunov = init_lyapunov + c / n * (gs[i] - phi[i])^2
    end


    set_initial_condition!(problem, init_lyapunov <= 1.0)


    final_lyapunov_avg = (xs - xs)^2
    for i in 1:n
        w = x0 + gamma * phi[i]
        for j in 1:n
            w = w - gamma / n * phi[j]
        end
        x1, gx1, _ = proximal_step!(w, fn[i], gamma)
        final_lyapunov = (xs - x1)^2
        for j in 1:n
            if i != j
                final_lyapunov = final_lyapunov + c / n * (phi[j] - gs[j])^2
            else
                final_lyapunov = final_lyapunov + c / n * (gs[j] - gx1)^2
            end
        end
        final_lyapunov_avg = final_lyapunov_avg + final_lyapunov / n
    end


    set_performance_metric!(problem, final_lyapunov_avg)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    kappa = mu * gamma / (1 + mu * gamma)
    theoretical_tau = (1 - kappa)


    if verbose
        println("*** Example file: worst-case performance of Point SAGA for a given Lyapunov function ***")
        println("\tPEPit guarantee:\t E[V(x^(1))] <= $(round(pepit_tau, digits=6)) V(x^(0))")
        println("\tTheoretical guarantee:\t E[V(x^(1))] <= $(round(theoretical_tau, digits=6)) V(x^(0))")
    end


    return pepit_tau, theoretical_tau
end


pepit_tau, theoretical_tau = wc_point_saga(1.0, 0.01, 10; solver=Clarabel.Optimizer, verbose=true)
