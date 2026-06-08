"""
    PEP()

Create an empty performance estimation problem and reset the global symbolic
registries used to index points, expressions, functions, constraints, and PSD
blocks.

A `PEP` stores the symbolic description of a worst-case analysis: function or
operator classes, initial points, initial conditions, performance metrics, and
optional PSD constraints. Calling [`solve!`](@ref) converts this symbolic
description into a semidefinite program in JuMP.

# Fields
- `list_of_functions`: function/operator classes declared in the problem.
- `list_of_points`: initial leaf points registered by [`set_initial_point!`](@ref).
- `list_of_conditions`: scalar initial or normalization constraints.
- `list_of_performance_metrics`: expressions whose minimum is maximized.
- `list_of_psd`: global PSD constraints.
- `residual`: dual residual matrix associated with the Gram PSD constraint.

# Mathematical model
PEPit computes the smallest valid worst-case constant by maximizing a
performance expression over all symbolic iterates and oracle values satisfying
the selected interpolation constraints and initial conditions.
"""
mutable struct PEP
    list_of_functions::Vector{AbstractFunction}
    list_of_points::Vector{Point}
    list_of_conditions::Vector{Constraint}
    list_of_performance_metrics::Vector{Expression}
    list_of_psd::Vector{PSDMatrix}
    residual::Union{Matrix{Float64},Nothing}

    function PEP()
        Point_counter[] = 0
        Expression_counter[] = 0
        Function_counter[] = 0
        Global_Constraint_counter[] = 0
        PSDMatrix_counter[] = 0
        BlockPartition_counter[] = 0
        NEXT_ID[] = 0

        empty!(GLOBAL_LEAF_POINTS)
        empty!(GLOBAL_LEAF_EXPRESSIONS)
        empty!(GLOBAL_BLOCK_PARTITIONS)

        return new([], [], [], [], [], nothing)
    end
end


"""
    declare_function!(pep, func_class, param; reuse_gradient=nothing)

Declare a leaf function or operator class in `pep` and return the created
object. `param` is typically an `OrderedDict` containing class parameters such
as `"L"`, `"mu"`, or `"beta"`.

# Arguments
- `pep::PEP`: problem to which the class is added.
- `func_class`: concrete subtype of [`AbstractFunction`](@ref).
- `param`: parameter dictionary consumed by `func_class`.
- `reuse_gradient`: optional override for repeated oracle evaluations at the
  same point.

# Examples
```julia
problem = PEP()
f = declare_function!(problem, SmoothConvexFunction, OrderedDict("L" => 1.0))
```
"""
function declare_function!(pep::PEP, func_class, param; reuse_gradient=nothing)
    f = reuse_gradient === nothing ?
        func_class(param; is_leaf=true) :
        func_class(param; is_leaf=true, reuse_gradient=reuse_gradient)
    push!(pep.list_of_functions, f)
    return f
end

"""
    declare_block_partition!(pep, d)

Create a [`BlockPartition`](@ref) with `d` blocks and register it for inclusion
in the PEP model.

The returned partition is global to the current `PEP` construction context.
Block orthogonality constraints are materialized during model construction.
"""
function declare_block_partition!(pep::PEP, d::Int)


    return BlockPartition(d)
end


"""
    add_constraint!(target, constraint)

Add a scalar [`Constraint`](@ref) to a PEP, function class, operator class, or
block partition, depending on `target`.

For a `PEP`, the constraint is treated as a global initial/general condition.
For a function-like object, it is treated as an additional class-specific
constraint and is included with that object's interpolation constraints.
"""
add_constraint!(pep::PEP, constraint::Constraint) = push!(pep.list_of_conditions, constraint)

"""
    set_initial_point!(pep)

Create and register a new leaf [`Point`](@ref) as an initial point of `pep`.

Initial points are independent vectors in the Gram matrix. They are typically
used in initial conditions such as `(x0 - xs)^2 <= R^2`.
"""
set_initial_point!(pep::PEP) = (x = Point(); push!(pep.list_of_points, x); x)

"""
    set_initial_condition!(pep, condition)

Add an initial condition or normalization constraint to `pep`.

Initial conditions define the admissible set of problem instances, for example
`\\|x_0 - x_\\star\\|^2 \\leq 1` or a bound on an initial function gap.
"""
set_initial_condition!(pep::PEP, condition::Constraint) = add_constraint!(pep, condition)

"""
    set_performance_metric!(pep, expression)

Register a scalar performance metric. `solve!` maximizes the minimum over all
registered metrics.

The expression is usually the quantity whose worst-case value is sought, such
as `f(x_n) - f_\\star`, `\\|x_n - x_\\star\\|^2`, or a residual norm. Multiple
metrics model the pointwise minimum of several quantities.
"""
set_performance_metric!(pep::PEP, expression::Expression) = push!(pep.list_of_performance_metrics, expression)


"""
    add_psd_matrix!(target, matrix_of_expressions)

Add a positive-semidefinite matrix constraint to a PEP or function-like object.
Entries may be [`Expression`](@ref) objects or real constants.

Global PSD constraints are attached to the problem. Function-local PSD
constraints are attached to the internal [`PEPFunction`](@ref) and compiled with
that class's interpolation constraints.
"""
function add_psd_matrix!(pep::PEP, matrix_of_expressions)
    push!(pep.list_of_psd, PSDMatrix(matrix_of_expressions))
    return pep.list_of_psd[end]
end


function _expression_to_jump(expr::Expression, F, G)

    pc = size(G, 1)
    ec = length(F)

    F_coeffs = zeros(Float64, ec)
    G_coeffs = zeros(Float64, pc, pc)
    const_val = 0.0

    if get_is_leaf(expr)
        @assert 0 <= expr.counter < ec "Expression counter out of bounds"
        F_coeffs[expr.counter+1] = 1.0
    else
        for (key, weight) in expr.decomposition_dict
            if key isa Expression
                @assert get_is_leaf(key)
                @assert 0 <= key.counter < ec "Expression counter out of bounds"
                F_coeffs[key.counter+1] += weight

            elseif key isa Tuple{Point,Point}
                p1, p2 = key
                @assert get_is_leaf(p1) && get_is_leaf(p2)
                @assert 0 <= p1.counter < pc && 0 <= p2.counter < pc "Point counter out of bounds"
                i, j = p1.counter + 1, p2.counter + 1
                G_coeffs[i, j] += weight

            elseif key == 1
                const_val += weight

            else
                error("Unsupported key in expression decomposition: $(typeof(key))")
            end
        end
    end

    G_coeffs .= 0.5 .* (G_coeffs .+ G_coeffs')

    sum_G = sum(G[i, j] * G_coeffs[i, j] for i in 1:pc, j in 1:pc)

    return const_val + dot(F_coeffs, F) + sum_G

end


function _eval_points_and_function_values!(pep::PEP, F_val::Vector{Float64}, G_val::Matrix{Float64}, verbose::Bool)
    ev = eigen(Symmetric(G_val))
    eig_val = ev.values
    eig_vec = ev.vectors
    if minimum(eig_val) < 0
        verbose && println("💻 PEPit:  Postprocessing: solver's output is not entirely feasible (smallest eigenvalue: $(minimum(eig_val)) < 0). Projecting Gram matrix.")
        eig_val = max.(eig_val, 0)
    end

    sqrt_diag = Diagonal(sqrt.(eig_val))
    points_values = qr(sqrt_diag * eig_vec').R

    for point_any in GLOBAL_LEAF_POINTS
        point = point_any::Point
        point._value = points_values[:, point.counter+1]
    end

    for expr_any in GLOBAL_LEAF_EXPRESSIONS
        expr = expr_any::Expression
        expr._value = F_val[expr.counter+1]
    end
end


function _apply_psd_duals!(packs)
    for (mat, psd_ref, eq_refs, sz) in packs
        mat._dual_variable_value = dual(psd_ref)
        n, m = sz
        @assert n == m
        entries = [dual(cref) for cref in eq_refs]
        mat.entries_dual_variable_value = reshape(collect(entries), n, m)
    end
end


function _eval_constraint_dual_values!(pep::PEP;
    perf_refs::Vector,
    init_refs::Vector,
    class_refs::Vector,
    main_psd_ref=nothing,
    global_psd_refs=Vector{Tuple}(),
    class_psd_refs=Vector{Tuple}())

    perf_duals = [dual(cr) for cr in perf_refs]
    pos_min = findmax(perf_duals)[2]

    if main_psd_ref !== nothing
        pep.residual = dual(main_psd_ref)
    else
        pep.residual = nothing
    end


    for (cond, cref) in zip(pep.list_of_conditions, init_refs)
        d = dual(cref)
        cond._dual_variable_value =
            cond.equality_or_inequality == "inequality" ? -d : d
    end


    idx = 1
    for f in pep.list_of_functions
        internal = _get_pep_func(f)
        for c in internal.list_of_constraints
            d = dual(class_refs[idx])
            c._dual_variable_value = c.equality_or_inequality == "inequality" ? -d : d
            idx += 1
        end
    end

    _apply_psd_duals!(global_psd_refs)
    _apply_psd_duals!(class_psd_refs)

    return pos_min
end


struct _PEPModelBuild
    model::Model
    variables::NamedTuple
    constraints::NamedTuple
    class_constraints::Vector{Constraint}
end

"""
    DualPEPCertificate

Store the solution of the explicit conic dual generated by [`solve_dual!`](@ref),
including scalar multipliers, PSD multipliers, model handles, and solver
statuses.

# Fields
- `dual_value`: objective value of the explicit dual model.
- `α`: multipliers for performance-metric epigraph constraints.
- `λ`: multipliers for inequality initial/general conditions.
- `ν`: multipliers for equality initial/general conditions.
- `θ`: multipliers for scalar interpolation constraints.
- `S`: dual residual matrix for the main Gram PSD constraint.
- `Y`: PSD multipliers for global and class-generated PSD blocks.
- `primal_model`, `dual_model`: JuMP model handles.
- `mappings`: symbolic objects associated with the multiplier arrays.
- `termination_status`, `primal_status`, `dual_status`: solver statuses.
"""
struct DualPEPCertificate
    dual_value::Float64
    α::Vector{Float64}
    λ::Vector{Float64}
    ν::Vector{Float64}
    θ::Vector{Float64}
    S::Matrix{Float64}
    Y::NamedTuple
    primal_model::Model
    dual_model::Model
    mappings::NamedTuple
    termination_status::Any
    primal_status::Any
    dual_status::Any
end

function _maybe_set_name!(ref, name_constraints::Bool, name::AbstractString)
    if name_constraints
        set_name(ref, String(name))
    end
    return ref
end

function _add_psd_matrix_block!(model::Model, psd_matrix::PSDMatrix, F, G;
    variable_base_name::AbstractString,
    constraint_base_name::AbstractString,
    name_constraints::Bool=false)

    n = psd_matrix.shape[1]
    M = @variable(model, [1:n, 1:n], Symmetric, base_name=String(variable_base_name))
    psd_ref = @constraint(model, M in PSDCone())
    _maybe_set_name!(psd_ref, name_constraints, "$(constraint_base_name)_psd")

    eq_refs = Vector{Any}()
    for i in 1:n, j in 1:n
        eq_ref = @constraint(model, M[i, j] == _expression_to_jump(psd_matrix[i, j], F, G))
        _maybe_set_name!(eq_ref, name_constraints, "$(constraint_base_name)_entry_$(i)_$(j)")
        push!(eq_refs, eq_ref)
    end

    return (psd_matrix, psd_ref, eq_refs, (n, n))
end

function _add_scalar_constraint!(model::Model, expr_jump, kind::String)
    if kind == "inequality"
        return @constraint(model, expr_jump <= 0)
    elseif kind == "equality"
        return @constraint(model, expr_jump == 0)
    else
        error("Constraint type must be either \"inequality\" or \"equality\". Got $(kind).")
    end
end

function _build_pep_jump_model!(pep::PEP;
    solver=Clarabel.Optimizer,
    verbose::Bool=true,
    name_constraints::Bool=false)

    for func in pep.list_of_functions
        add_class_constraints!(func)
    end

    model = Model(solver)
    if !verbose
        set_silent(model)
    end

    pc, ec = Point_counter[], Expression_counter[]
    verbose && println(" 💻 PEPit:  Setting up the problem: size of the main PSD matrix: $(pc)x$(pc)")

    @variable(model, objective)
    @variable(model, F[1:ec])
    @variable(model, G[1:pc, 1:pc], Symmetric)
    main_psd_ref = @constraint(model, G in PSDCone())
    _maybe_set_name!(main_psd_ref, name_constraints, "main_gram_psd")

    verbose && println(" 💻 PEPit:  Setting up the problem: performance measure is minimum of $(length(pep.list_of_performance_metrics)) element(s)")
    perf_con_refs = Vector{Any}()
    for (r, metric) in enumerate(pep.list_of_performance_metrics)
        con = @constraint(model, objective <= _expression_to_jump(metric, F, G))
        _maybe_set_name!(con, name_constraints, "alpha_metric_$(r)")
        push!(perf_con_refs, con)
    end

    verbose && println(" 💻 PEPit:  Setting up the problem: Adding initial conditions and general constraints ...")
    initial_con_refs = Vector{Any}()
    for (ell, cond) in enumerate(pep.list_of_conditions)
        expr_jump = _expression_to_jump(cond.expression, F, G)
        cref = _add_scalar_constraint!(model, expr_jump, cond.equality_or_inequality)
        prefix = cond.equality_or_inequality == "inequality" ? "lambda" : "nu"
        _maybe_set_name!(cref, name_constraints, "$(prefix)_condition_$(ell)")
        push!(initial_con_refs, cref)
    end
    verbose && println(" 💻 PEPit:  Setting up the problem: initial conditions and general constraints ($(length(pep.list_of_conditions)) constraint(s) added)")


    partition_con_refs = Vector{Any}()
    for (bpi, bp) in enumerate(GLOBAL_BLOCK_PARTITIONS)
        add_partition_constraints!(bp)
        for (m, c) in enumerate(bp.list_of_constraints)
            expr_jump = _expression_to_jump(c.expression, F, G)
            cref = _add_scalar_constraint!(model, expr_jump, c.equality_or_inequality)
            _maybe_set_name!(cref, name_constraints, "partition_$(bpi)_constraint_$(m)")
            push!(partition_con_refs, cref)
        end
    end
    if !isempty(GLOBAL_BLOCK_PARTITIONS)
        verbose && println(" 💻 PEPit:  Setting up the problem: $(length(partition_con_refs)) block-partition orthogonality constraint(s) added")
    end

    global_psd_refs = Vector{Tuple}()
    if !isempty(pep.list_of_psd)
        verbose && println(" 💻 PEPit:  Setting up the problem: $(length(pep.list_of_psd)) lmi constraint(s) added")
        for (k, psd_matrix) in enumerate(pep.list_of_psd)
            pack = _add_psd_matrix_block!(model, psd_matrix, F, G;
                variable_base_name="problem_psd_$(k)",
                constraint_base_name="Y_problem_$(k)",
                name_constraints=name_constraints)
            push!(global_psd_refs, pack)
            verbose && println("\t\t Size of PSD matrix $(k): $(psd_matrix.shape[1])x$(psd_matrix.shape[1])")
        end
    end

    verbose && println(" 💻 PEPit:  Setting up the problem: interpolation conditions for $(length(pep.list_of_functions)) function(s)")
    class_con_refs = Vector{Any}()
    class_constraints = Constraint[]
    class_psd_refs = Vector{Tuple}()
    for (i, f) in enumerate(pep.list_of_functions)
        internal = _get_pep_func(f)

        added = 0
        for c in internal.list_of_constraints
            expr_jump = _expression_to_jump(c.expression, F, G)
            cref = _add_scalar_constraint!(model, expr_jump, c.equality_or_inequality)
            _maybe_set_name!(cref, name_constraints, "theta_function_$(i)_constraint_$(added + 1)")
            push!(class_con_refs, cref)
            push!(class_constraints, c)
            added += 1
        end
        verbose && println("\t\t function $i : $added scalar constraint(s) added")

        if !isempty(internal.list_of_class_psd)
            verbose && println("\t\t function $i : Adding $(length(internal.list_of_class_psd)) lmi constraint(s) ...")
            for (k, psd_matrix) in enumerate(internal.list_of_class_psd)
                pack = _add_psd_matrix_block!(model, psd_matrix, F, G;
                    variable_base_name="class_psd_$(i)_$(k)",
                    constraint_base_name="Y_class_$(i)_$(k)",
                    name_constraints=name_constraints)
                push!(class_psd_refs, pack)
                verbose && println("\t\t function $i : size of PSD matrix $(k): $(psd_matrix.shape[1])x$(psd_matrix.shape[1])")
            end
            verbose && println("\t\t function $i : $(length(internal.list_of_class_psd)) lmi constraint(s) added")
        end

        if !isempty(internal.list_of_psd)
            verbose && println("\t\t function $i : Adding $(length(internal.list_of_psd)) lmi constraint(s) ...")
            for (k, psd_matrix) in enumerate(internal.list_of_psd)
                pack = _add_psd_matrix_block!(model, psd_matrix, F, G;
                    variable_base_name="function_psd_$(i)_$(k)",
                    constraint_base_name="Y_function_$(i)_$(k)",
                    name_constraints=name_constraints)
                push!(class_psd_refs, pack)
                verbose && println("\t\t function $i : size of PSD matrix $(k): $(psd_matrix.shape[1])x$(psd_matrix.shape[1])")
            end
            verbose && println("\t\t function $i : $(length(internal.list_of_psd)) lmi constraint(s) added")
        end
    end

    verbose && println(" 💻 PEPit:  Compiling SDP")
    @objective(model, Max, objective)

    return _PEPModelBuild(
        model,
        (objective=objective, F=F, G=G),
        (performance=perf_con_refs,
            initial=initial_con_refs,
            class=class_con_refs,
            main_psd=main_psd_ref,
            global_psd=global_psd_refs,
            class_psd=class_psd_refs),
        class_constraints)
end

function _single_dual_variable_value(dual_model::Model, primal_constraint_ref)
    vars = Dualization._get_dual_variables(dual_model, primal_constraint_ref)
    vars === nothing && error("Dualization did not expose a scalar dual variable for primal constraint $(primal_constraint_ref).")
    length(vars) == 1 || error("Expected one scalar dual variable for primal constraint $(primal_constraint_ref), got $(length(vars)).")
    return Float64(value(vars[1]))
end

function _write_scalar_duals_from_dual_model!(dual_model::Model, constraints::Vector{Constraint}, refs::Vector)
    values = Float64[]
    length(constraints) == length(refs) || error("Dual extraction mismatch: $(length(constraints)) PEP constraints but $(length(refs)) JuMP references.")
    for (constraint, cref) in zip(constraints, refs)
        raw = _single_dual_variable_value(dual_model, cref)
        normalized = constraint.equality_or_inequality == "inequality" ? -raw : raw
        constraint._dual_variable_value = normalized
        push!(values, normalized)
    end
    return values
end

function _symmetric_dual_matrix_from_variable_map(dual_model::Model, X)
    n = size(X, 1)
    S = zeros(Float64, n, n)
    values_by_constraint = IdDict{Any, Vector{Float64}}()

    for i in 1:n, j in i:n
        dual_constraint_ref, idx = Dualization._get_dual_constraint(dual_model, X[i, j])
        dual_constraint_ref === nothing && error("Dualization did not expose a dual constraint for a PSD variable entry.")
        vals = get!(values_by_constraint, dual_constraint_ref) do
            Float64.(value.(constraint_object(dual_constraint_ref).func))
        end
        S[i, j] = vals[idx]
        S[j, i] = vals[idx]
    end

    return 0.5 .* (S .+ S')
end

function _apply_psd_duals_from_dual_model!(dual_model::Model, packs)
    Y_values = Matrix{Float64}[]
    for (mat, _psd_ref, eq_refs, sz) in packs
        n, m = sz
        @assert n == m
        entries = [_single_dual_variable_value(dual_model, cref) for cref in eq_refs]
        mat.entries_dual_variable_value = reshape(collect(entries), n, m)
        mat._dual_variable_value = -0.5 .* (mat.entries_dual_variable_value .+ mat.entries_dual_variable_value')
        push!(Y_values, mat._dual_variable_value)
    end
    return Y_values
end

"""
    solve_dual!(pep; solver=Clarabel.Optimizer, verbose=true)

Build the primal SDP for `pep`, dualize it with `Dualization.jl`, solve the
explicit dual model, and return a [`DualPEPCertificate`](@ref).

This routine is useful when the dual multipliers themselves are part of the
output, for example when reconstructing a proof certificate for a worst-case
bound. It also writes scalar and PSD dual values back to the symbolic
constraints so that [`eval_dual`](@ref) can be used afterwards.

# Arguments
- `pep::PEP`: symbolic performance estimation problem.
- `solver`: JuMP optimizer constructor used for the dual model.
- `verbose`: print model-building and solver progress when true.
"""
function solve_dual!(pep::PEP;
    solver=Clarabel.Optimizer,
    verbose::Bool=true)

    build = _build_pep_jump_model!(pep;
        solver=solver,
        verbose=verbose,
        name_constraints=true)

    verbose && println(" 💻 PEPit:  Dualizing SDP")
    dual_model = Dualization.dualize(build.model, solver)
    if !verbose
        set_silent(dual_model)
    end

    verbose && println(" 💻 PEPit:  Calling SDP solver on the explicit dual")
    optimize!(dual_model)
    if verbose
        println(" 💻 PEPit:  Dual solver status: $(termination_status(dual_model)); optimal value: $(objective_value(dual_model))")
    end

    α = [-_single_dual_variable_value(dual_model, cref) for cref in build.constraints.performance]
    λν_constraints = pep.list_of_conditions
    scalar_condition_values = _write_scalar_duals_from_dual_model!(dual_model, λν_constraints, build.constraints.initial)
    λ = [value for (value, c) in zip(scalar_condition_values, λν_constraints) if c.equality_or_inequality == "inequality"]
    ν = [value for (value, c) in zip(scalar_condition_values, λν_constraints) if c.equality_or_inequality == "equality"]
    θ = _write_scalar_duals_from_dual_model!(dual_model, build.class_constraints, build.constraints.class)

    S = _symmetric_dual_matrix_from_variable_map(dual_model, build.variables.G)
    pep.residual = S

    Y_problem = _apply_psd_duals_from_dual_model!(dual_model, build.constraints.global_psd)
    Y_class = _apply_psd_duals_from_dual_model!(dual_model, build.constraints.class_psd)

    return DualPEPCertificate(
        Float64(objective_value(dual_model)),
        α,
        λ,
        ν,
        θ,
        S,
        (problem=Y_problem, class=Y_class),
        build.model,
        dual_model,
        (α=pep.list_of_performance_metrics,
            λ=[c for c in λν_constraints if c.equality_or_inequality == "inequality"],
            ν=[c for c in λν_constraints if c.equality_or_inequality == "equality"],
            θ=build.class_constraints,
            S=build.constraints.main_psd,
            Y=(problem=[pack[1] for pack in build.constraints.global_psd],
                class=[pack[1] for pack in build.constraints.class_psd])),
        termination_status(dual_model),
        primal_status(dual_model),
        dual_status(dual_model))
end


function _get_nb_eigs_and_corrected(M::AbstractMatrix{<:Real})
    S = 0.5 .* (M .+ M')
    ev = eigen(Symmetric(S))
    λ = ev.values
    V = ev.vectors
    maxpos = maximum(λ)
    maxneg = -minimum(λ)
    eig_threshold = max(maxpos / 1e3, 2 * maxneg)
    nonzero = λ .>= eig_threshold
    nb = count(nonzero)
    λcorr = nonzero .* λ
    Scorr = V * Diagonal(λcorr) * V'
    t = nb < length(λ) ? max(maximum(λ[.!nonzero]), 0.0) : 0.0
    return nb, t, Scorr
end


function _logdet_dimension_reduction!(model::JuMP.Model, G, objective, wc_value::Float64;
    niter::Int, eig_regularization::Float64,
    tol::Float64, verbose::Bool)

    pc = size(G, 1)

    Gval = value.(G)
    _, _, Gcorr = _get_nb_eigs_and_corrected(Gval)

    @constraint(model, objective >= wc_value - tol)

    for k in 1:niter
        W = inv(Symmetric(Gcorr + eig_regularization * I(pc)))

        @objective(model, Min, sum(W[i, j] * G[i, j] for i in 1:pc, j in 1:pc))
        verbose && println(" 💻 PEPit:  Calling SDP solver (logdet step $k)")
        optimize!(model)

        wc_value = value(objective)

        Gval = value.(G)
        nb2, thr2, Gcorr = _get_nb_eigs_and_corrected(Gval)

        if verbose
            println(" 💻 PEPit:  Solver status: $(termination_status(model)); objective value: $(wc_value)")
            println(" 💻 PEPit:  Postprocessing: $nb2 eigenvalue(s) > $thr2 after $k logdet step(s)")
        end
    end

    return wc_value

end


"""
    solve!(pep; solver=Clarabel.Optimizer, verbose=true, tracetrick=false,
           logdetiters=0, eig_regularization=1e-3,
           tol_dimension_reduction=1e-5, return_full_model=false)

Build and solve the primal SDP associated with `pep`. Return the worst-case
value unless `return_full_model=true`, in which case solver variables,
constraints, and residual data are returned as a named tuple.

The constructed SDP uses a Gram matrix for all leaf [`Point`](@ref) objects and
one scalar variable for each leaf [`Expression`](@ref). Initial conditions,
performance metrics, interpolation constraints, and PSD blocks are translated
into JuMP constraints before the model is optimized.

# Arguments
- `pep::PEP`: symbolic problem to solve.
- `solver`: JuMP optimizer constructor, for example `Clarabel.Optimizer`.
- `verbose`: print model-building and solver progress when true.
- `tracetrick`: run a trace-minimization heuristic after the first solve.
- `logdetiters`: number of log-det heuristic iterations for dimension
  reduction.
- `eig_regularization`: regularization used in the log-det heuristic.
- `tol_dimension_reduction`: allowed objective degradation during dimension
  reduction.
- `return_full_model`: return JuMP model data instead of only the worst-case
  value.

# Returns
The worst-case value as a `Float64`, or a named tuple with model internals when
`return_full_model=true`.

See also [`solve_dual!`](@ref), [`evaluate`](@ref), and [`eval_dual`](@ref).
"""
function solve!(pep::PEP;
    solver=Clarabel.Optimizer,
    verbose::Bool=true,
    tracetrick::Bool=false,
    logdetiters::Int=0,
    eig_regularization::Float64=1e-3,
    tol_dimension_reduction::Float64=1e-5,
    return_full_model::Bool=false,
)

    build = _build_pep_jump_model!(pep; solver=solver, verbose=verbose)
    model = build.model
    objective = build.variables.objective
    F = build.variables.F
    G = build.variables.G
    pc = Point_counter[]
    perf_con_refs = build.constraints.performance
    initial_con_refs = build.constraints.initial
    class_con_refs = build.constraints.class
    main_psd_ref = build.constraints.main_psd
    global_psd_refs = build.constraints.global_psd
    class_psd_refs = build.constraints.class_psd


    verbose && println(" 💻 PEPit:  Calling SDP solver")
    optimize!(model)
    if verbose
        println(" 💻 PEPit:  Solver status: $(termination_status(model)); optimal value: $(objective |> value)")
    end
    wc_value = value(objective)


    if tracetrick
        tol = tol_dimension_reduction
        @constraint(model, objective >= wc_value - tol)
        @objective(model, Min, sum(G[i, i] for i in 1:pc))
        verbose && println(" 💻 PEPit:  Calling SDP solver (trace heuristic)")
        optimize!(model)
        wc_value = value(objective)
        if verbose
            println(" 💻 PEPit:  Solver status: $(termination_status(model)); objective value: $(wc_value)")
        end
    end


    if logdetiters > 0
        nb, thr, _ = _get_nb_eigs_and_corrected(value.(G))
        if verbose
            println(" 💻 PEPit:  Postprocessing: $nb eigenvalue(s) > $thr before dimension reduction")
        end
        wc_value = _logdet_dimension_reduction!(model, G, objective, wc_value;
            niter=logdetiters, eig_regularization=eig_regularization,
            tol=tol_dimension_reduction, verbose=verbose)
    end


    F_val = value.(F)
    G_val = value.(G)
    _eval_points_and_function_values!(pep, F_val, G_val, verbose)


    pos_min_metric = _eval_constraint_dual_values!(pep;
        perf_refs=perf_con_refs,
        init_refs=initial_con_refs,
        class_refs=class_con_refs,
        main_psd_ref=main_psd_ref,
        global_psd_refs=global_psd_refs,
        class_psd_refs=class_psd_refs)


    if return_full_model
        return (wc_value=wc_value,
            model=model,
            variables=(objective=objective, F=F, G=G),
            constraints=(performance=perf_con_refs,
                initial=initial_con_refs,
                class=class_con_refs,
                main_psd=main_psd_ref,
                global_psd=global_psd_refs,
                class_psd=class_psd_refs),
            position_of_min_metric=pos_min_metric,
            residual=pep.residual)
    else
        return wc_value
    end
end
