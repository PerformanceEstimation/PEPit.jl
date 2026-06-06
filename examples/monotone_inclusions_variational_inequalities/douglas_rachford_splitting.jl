using PEPit, OrderedCollections, Clarabel

function wc_douglas_rachford_splitting(L, mu, alpha, theta; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, LipschitzStronglyMonotoneOperatorCheap, OrderedDict("L" => L, "mu" => 0))
    B = declare_function!(problem, StronglyMonotoneOperator, OrderedDict("mu" => mu))


    w0 = set_initial_point!(problem)
    w1 = set_initial_point!(problem)


    set_initial_condition!(problem, (w0 - w1)^2 <= 1)


    x0, _, _ = proximal_step!(w0, B, alpha)
    y0, _, _ = proximal_step!(2 * x0 - w0, A, alpha)
    z0 = w0 - theta * (x0 - y0)


    x1, _, _ = proximal_step!(w1, B, alpha)
    y1, _, _ = proximal_step!(2 * x1 - w1, A, alpha)
    z1 = w1 - theta * (x1 - y1)


    set_performance_metric!(problem, (z0 - z1)^2)


    pepit_tau = solve!(problem; solver=solver, verbose=verbose)


    mu = alpha * mu
    L = alpha * L
    c = sqrt(((2 * (theta - 1) * mu + theta - 2)^2 + L^2 * (theta - 2 * (mu + 1))^2) / (L^2 + 1))
    if theta * (theta + c) / (mu + 1)^2 / c * (
            c + mu * ((2 * (theta - 1) * mu + theta - 2) - L^2 * (theta - 2 * (mu + 1))) / (L^2 + 1)) >= 0
        theoretical_tau = ((theta + c) / 2 / (mu + 1))^2
    elseif (L <= 1) && (mu >= (L^2 + 1) / (L - 1)^2) && (theta <= -(2 * (mu + 1) * (L + 1) *
                                                                    (mu + (mu - 1) * L^2 - 2 * mu * L - 1)) / (
                                                             mu + L * (L^2 + L + 1) + 2 * mu^2 * (L - 1)
                                                             + mu * L * (1 - (L - 3) * L) + 1))
        theoretical_tau = (1 - theta * (L + mu) / (L + 1) / (mu + 1))^2
    else
        theoretical_tau = (2 - theta) / 4 / mu / (L^2 + 1) * (
            theta * (1 - 2 * mu + L^2) - 2 * mu * (L^2 - 1)) *
                          (theta * (1 + 2 * mu + L^2) - 2 * (mu + 1) * (L^2 + 1)) / (
            theta * (1 + 2 * mu - L^2) - 2 * (mu + 1) * (1 - L^2))
    end


    if verbose
        println("*** Example file: worst-case performance of the Douglas Rachford Splitting***")
        println("\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(pepit_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
        println("\tTheoretical guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(theoretical_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_douglas_rachford_splitting(1.0, 0.1, 1.3, 0.9; verbose=true)
