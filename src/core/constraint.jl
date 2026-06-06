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


eval_dual(c::Constraint) = isnothing(c._dual_variable_value) ? error("PEP must be solved") : c._dual_variable_value
