"""
    Constraint(expression, equality_or_inequality)

Represent a scalar equality or inequality constraint in a PEP.

The stored `expression` is the canonical left-hand side. Inequalities are stored
in the form `expression <= 0`, while equalities are stored as `expression == 0`.
Users normally create constraints through overloaded comparisons such as
`expr <= 1`, `expr1 >= expr2`, or `expr1 == expr2`.

# Fields
- `expression`: symbolic scalar residual.
- `equality_or_inequality`: either `"equality"` or `"inequality"`.
- `counter`: global scalar-constraint index.
- `_dual_variable_value`: dual multiplier populated after solving.
- `_value`: numerical residual populated by [`evaluate`](@ref).

# Mathematical model
Scalar constraints encode initial conditions, performance metric epigraph
constraints, function/operator interpolation inequalities, and auxiliary
relations introduced by primitive steps.

See also [`Expression`](@ref), [`add_constraint!`](@ref),
[`set_initial_condition!`](@ref), and [`eval_dual`](@ref).
"""
mutable struct Constraint <: AbstractConstraint
    expression::Expression
    equality_or_inequality::String
    counter::Int
    _dual_variable_value::Union{Float64,Nothing}
    _value::Union{Float64,Nothing}

    function Constraint(expression::Expression, equality_or_inequality::String)
        @assert equality_or_inequality in ["equality", "inequality"]
        counter = Global_Constraint_counter[]
        Global_Constraint_counter[] += 1
        return new(expression, equality_or_inequality, counter, nothing, nothing)
    end
end


Base.:(<=)(e1::Expression, e2::Expression) = Constraint(e1 - e2, "inequality")

Base.:(<=)(e1::Expression, e2::Real)       = Constraint(e1 - e2, "inequality")

Base.:(<=)(e1::Real,       e2::Expression) = Constraint(e1 - e2, "inequality")

Base.:(>=)(e1::Expression, e2::Expression) = Constraint(e2 - e1, "inequality")

Base.:(>=)(e1::Expression, e2::Real)       = Constraint(e2 - e1, "inequality")

Base.:(>=)(e1::Real,       e2::Expression) = Constraint(e2 - e1, "inequality")

Base.:(==)(e1::Expression, e2::Real)       = Constraint(e1 - e2, "equality")

Base.:(==)(e1::Real,       e2::Expression) = Constraint(Expression(e1) - e2, "equality")


"""
    evaluate(c::Constraint)

Return the numerical residual of `c.expression` after the PEP has been solved.

For inequalities, feasibility corresponds to a nonpositive residual because the
canonical form is `expression <= 0`. For equalities, feasibility corresponds to
a residual close to zero.
"""
function evaluate(c::Constraint)
    if isnothing(c._value)
        try
            c._value = evaluate(c.expression)
        catch err

            error("The PEP must be solved to evaluate Constraints!")
        end
    end
    return c._value
end


"""
    eval_dual(obj)

Return the dual multiplier associated with a scalar or PSD constraint after
[`solve!`](@ref) or [`solve_dual!`](@ref) has populated dual values.

For a scalar [`Constraint`](@ref), the return value is a number. For a
[`PSDMatrix`](@ref), the return value is the corresponding symmetric dual
matrix.
"""
eval_dual(c::Constraint) = isnothing(c._dual_variable_value) ? error("PEP must be solved") : c._dual_variable_value
