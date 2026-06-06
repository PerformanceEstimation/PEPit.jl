using OrderedCollections

mutable struct NonexpansiveOperator <: AbstractFunction
    v::Union{Point,Nothing}
    _PEPit_func::PEPFunction

    function NonexpansiveOperator(param=OrderedDict(); is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=true)
        v = haskey(param, "v") ? param["v"] : nothing
        return new(v, func)
    end
end

gradient!(op::NonexpansiveOperator, p::Point) = gradient!(op._PEPit_func, p)
value!(op::NonexpansiveOperator, p::Point) = value!(op._PEPit_func, p)
stationary_point!(op::NonexpansiveOperator) = stationary_point!(op._PEPit_func)
add_constraint!(op::NonexpansiveOperator, constraint::Constraint) = add_constraint!(op._PEPit_func, constraint)

function add_class_constraints!(op::NonexpansiveOperator)
    pts = op._PEPit_func.list_of_points


    for i in 1:length(pts), j in (i + 1):length(pts)
        xi, gi, _ = pts[i]
        xj, gj, _ = pts[j]
        add_constraint!(op, (gi - gj)^2 - (xi - xj)^2 <= 0)
    end


    if op.v !== nothing
        for (xi, gi, _) in pts
            add_constraint!(op, op.v^2 - (xi - gi) * op.v <= 0)
        end
    end
end

_get_pep_func(op::NonexpansiveOperator) = op._PEPit_func
