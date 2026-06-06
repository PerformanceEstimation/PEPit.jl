using PEPit, OrderedCollections, Clarabel

function wc_douglas_rachford_splitting_2(beta, mu, alpha, theta; solver=Clarabel.Optimizer, verbose=true)


    problem = PEP()


    A = declare_function!(problem, CocoerciveOperator, OrderedDict("beta" => beta))
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
    beta = alpha * beta
    if mu * beta - mu + beta < 0 && theta <= 2 * (beta + 1) * (mu - beta - mu * beta) /
                                            (mu + mu * beta - beta - beta^2 - 2 * mu * beta^2)
        theoretical_tau = (1 - theta * beta / (beta + 1))^2
    elseif mu * beta - mu - beta > 0 && theta <= 2 * (mu^2 + beta^2 + mu * beta + mu + beta - mu^2 * beta^2) /
                                                 (mu^2 + beta^2 + mu^2 * beta + mu * beta^2 + mu + beta - 2 * mu^2 * beta^2)
        theoretical_tau = (1 - theta * (1 + mu * beta) / (mu + 1) / (beta + 1))^2
    elseif theta >= 2 * (mu * beta + mu + beta) / (2 * mu * beta + mu + beta)
        theoretical_tau = (1 - theta)^2
    elseif mu * beta + mu - beta < 0 && theta <= 2 * (mu + 1) * (beta - mu - mu * beta) /
                                                 (beta + mu * beta - mu - mu^2 - 2 * mu^2 * beta)
        theoretical_tau = (1 - theta * mu / (mu + 1))^2
    else
        theoretical_tau = (2 - theta) / 4 / mu * ((2 - theta) * mu * (beta + 1) + theta * beta * (1 - mu)) *
                          ((2 - theta) * beta * (mu + 1) + theta * mu * (1 - beta)) / mu / beta /
                          (2 * mu * beta * (1 - theta) + (2 - theta) * (mu + beta + 1))
    end


    if verbose != -1
        println("*** Example file: worst-case performance of the Douglas Rachford Splitting***")
        println("\tPEPit guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(pepit_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
        println("\tTheoretical guarantee:\t ||w_(t+1)^0 - w_(t+1)^1||^2 <= $(round(theoretical_tau, digits=6)) ||w_(t)^0 - w_(t)^1||^2")
    end


    return pepit_tau, theoretical_tau
end

pepit_tau, theoretical_tau =
    wc_douglas_rachford_splitting_2(1.2, 0.1, 0.3, 1.5; verbose=true)
