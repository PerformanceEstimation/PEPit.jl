"""
    PSDMatrix(matrix_of_expressions)

Represent a positive-semidefinite matrix constraint with symbolic entries.

A `PSDMatrix` stores a square matrix whose entries are [`Expression`](@ref)
objects or real constants. It is used for interpolation conditions and
additional matrix inequalities that are more naturally expressed as PSD blocks
than as scalar inequalities.

# Fields
- `matrix_of_expressions`: square matrix of symbolic scalar entries.
- `shape`: matrix dimensions.
- `_value`: numerical matrix value after solving.
- `_dual_variable_value`: PSD dual matrix after solving.
- `entries_dual_variable_value`: duals associated with entry-linking
  equalities in the JuMP model.
- `counter`: global PSD-constraint index.

See also [`add_psd_matrix!`](@ref), [`evaluate`](@ref), and [`eval_dual`](@ref).
"""
mutable struct PSDMatrix
    matrix_of_expressions::Matrix{Expression}
    shape::Tuple{Int,Int}
    _value::Union{Matrix{Float64},Nothing}
    _dual_variable_value::Union{Matrix{Float64},Nothing}
    entries_dual_variable_value::Union{Matrix{Float64},Nothing}
    counter::Int

    function PSDMatrix(matrix_of_expressions)

        local_counter = PSDMatrix_counter[]
        PSDMatrix_counter[] += 1


        stored = _store_matrix_of_expressions(matrix_of_expressions)
        shp = size(stored)

        return new(stored, shp, nothing, nothing, nothing, local_counter)
    end
end


PSDMatrix(; matrix_of_expressions) = PSDMatrix(matrix_of_expressions)


function _store_matrix_of_expressions(matrix_of_expressions)


    mat_any = (
        if matrix_of_expressions isa AbstractMatrix
            n1, n2 = size(matrix_of_expressions)
            tmp = Array{Any}(undef, n1, n2)
            for i in 1:n1, j in 1:n2
                tmp[i, j] = matrix_of_expressions[i, j]
            end
            tmp
        elseif matrix_of_expressions isa AbstractVector
            n1 = length(matrix_of_expressions)
            @assert n1 > 0 "PSDMatrix requires a non-empty square matrix"
            n2 = length(matrix_of_expressions[1])
            @assert all(length(row) == n2 for row in matrix_of_expressions) "Rows must have equal length"
            tmp = Array{Any}(undef, n1, n2)
            for i in 1:n1, j in 1:n2
                tmp[i, j] = matrix_of_expressions[i][j]
            end
            tmp
        else
            error("PSDMatrix expects a square matrix (AbstractMatrix or Vector of Vectors). Got $(typeof(matrix_of_expressions))")
        end
    )


    n, m = size(mat_any)
    @assert n == m "PSDMatrix requires a square matrix. Got $(n)x$(m)."


    mat_expr = Array{Expression}(undef, n, n)
    for i in 1:n, j in 1:n
        v = mat_any[i, j]
        if v isa Expression
            mat_expr[i, j] = v
        elseif v isa Real

            mat_expr[i, j] = Expression(v)
        else
            error("PSD matrices contain only Expressions and/or scalar values! Got $(typeof(v)).")
        end
    end

    return mat_expr
end


Base.getindex(psd::PSDMatrix, i::Int, j::Int) = psd.matrix_of_expressions[i, j]


"""
    evaluate(psd::PSDMatrix)

Return the numerical matrix value of a PSD constraint after the PEP has been
solved.

Each symbolic entry is evaluated independently. Calling this before the PEP has
been solved raises an error.
"""
function evaluate(psd::PSDMatrix)
    if psd._value === nothing
        try

            psd._value = map(ex -> evaluate(ex), psd.matrix_of_expressions)
        catch err
            error("The PEP must be solved to evaluate PSDMatrix!")
        end
    end
    return psd._value
end


function eval_dual(psd::PSDMatrix)
    if psd._dual_variable_value === nothing
        error("The PEP must be solved to evaluate PSDMatrix dual variables!")
    end
    return psd._dual_variable_value
end
