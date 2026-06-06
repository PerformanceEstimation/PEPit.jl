module PEPit


using JuMP
using Dualization
using Mosek, MosekTools, Clarabel
using LinearAlgebra
using OrderedCollections
using OffsetArrays
import Base: +, -, *, /, ==, <=, >=, ^, hash, getindex


abstract type AbstractPoint end
abstract type AbstractExpression end
abstract type AbstractConstraint end
abstract type AbstractFunction end


const GLOBAL_LEAF_POINTS = Vector{Any}()
const GLOBAL_LEAF_EXPRESSIONS = Vector{Any}()


const Point_counter = Ref(0)
const Expression_counter = Ref(0)
const Function_counter = Ref(0)
const Global_Constraint_counter = Ref(0)
const NEXT_ID = Ref(0)
const PSDMatrix_counter = Ref(0)
const BlockPartition_counter = Ref(0)


const GLOBAL_BLOCK_PARTITIONS = Vector{Any}()


include("tools/dict_operations.jl")


include("core/point.jl")
include("core/expression.jl")
include("core/constraint.jl")
include("core/block_partition.jl")
include("core/psd_matrix.jl")
include("core/function.jl")


include("functions/convex_function.jl")
include("functions/convex_lipschitz_function.jl")
include("functions/smooth_function.jl")
include("functions/smooth_convex_function.jl")
include("functions/smooth_strongly_convex_function.jl")
include("functions/strongly_convex_function.jl")
include("functions/convex_indicator.jl")
include("functions/convex_qg_function.jl")
include("functions/convex_support_function.jl")
include("functions/rsi_eb_function.jl")
include("functions/smooth_convex_lipschitz_function.jl")
include("functions/smooth_strongly_convex_quadratic_function.jl")
include("functions/smooth_quadratic_lojasiewicz_function_cheap.jl")
include("functions/smooth_quadratic_lojasiewicz_function_expensive.jl")
include("functions/block_smooth_convex_function_cheap.jl")
include("functions/block_smooth_convex_function_expensive.jl")


include("operators/lipschitz.jl")
include("operators/linear.jl")
include("operators/nonexpansive.jl")
include("operators/monotone.jl")
include("operators/strongly_monotone.jl")
include("operators/negatively_comonotone.jl")
include("operators/cocoercive.jl")
include("operators/cocoercive_strongly_monotone_cheap.jl")
include("operators/cocoercive_strongly_monotone_expensive.jl")
include("operators/lipschitz_strongly_monotone_cheap.jl")
include("operators/lipschitz_strongly_monotone_expensive.jl")
include("operators/symmetric_linear.jl")
include("operators/skew_symmetric_linear.jl")


include("core/pep.jl")


include("primitive_steps/inexact_gradient_step.jl")
include("primitive_steps/bregman_gradient_step.jl")
include("primitive_steps/bregman_proximal_step.jl")
include("primitive_steps/epsilon_subgradient_step.jl")
include("primitive_steps/exact_linesearch_step.jl")
include("primitive_steps/inexact_proximal_step.jl")
include("primitive_steps/proximal_step.jl")
include("primitive_steps/linear_optimization_step.jl")
include("primitive_steps/shifted_optimization_step.jl")


export

    PEP, Point, Expression, Constraint, PSDMatrix,

    BlockPartition, declare_block_partition!, get_block, get_nb_blocks,

    AbstractFunction, PEPFunction,
    ConvexFunction, ConvexLipschitzFunction, SmoothFunction, SmoothConvexFunction, SmoothStronglyConvexFunction, StronglyConvexFunction, ConvexIndicatorFunction,
    ConvexQGFunction, ConvexSupportFunction, RsiEbFunction, SmoothConvexLipschitzFunction, SmoothStronglyConvexQuadraticFunction,
    SmoothQuadraticLojasiewiczFunctionCheap, SmoothQuadraticLojasiewiczFunctionExpensive,
    BlockSmoothConvexFunctionCheap, BlockSmoothConvexFunctionExpensive,

    LipschitzOperator, LinearOperator, NonexpansiveOperator, MonotoneOperator,
    StronglyMonotoneOperator, NegativelyComonotoneOperator, CocoerciveOperator,
    CocoerciveStronglyMonotoneOperatorCheap, CocoerciveStronglyMonotoneOperatorExpensive,
    LipschitzStronglyMonotoneOperatorCheap, LipschitzStronglyMonotoneOperatorExpensive,
    SymmetricLinearOperator, SkewSymmetricLinearOperator,

    solve!, solve_dual!, DualPEPCertificate,
    declare_function!, set_initial_point!, set_initial_condition!,
    set_performance_metric!, add_constraint!, add_psd_matrix!,

    oracle!, gradient!, value!, stationary_point!, fixed_point!,

    inexact_gradient_step!, bregman_gradient_step!, bregman_proximal_step!, epsilon_subgradient_step!, exact_linesearch_step!, inexact_proximal_step!, proximal_step!, linear_optimization_step!, shifted_optimization_step!,

    merge_dicts, multiply_dicts, prune_dict,

    Point_counter, Expression_counter, Function_counter,
    Global_Constraint_counter, NEXT_ID, PSDMatrix_counter,

    get_is_leaf,
    _is_already_evaluated_on_point, _separate_leaf_functions_regarding_their_need_on_point,
    _get_nb_eigs_and_corrected,

    eval_dual

end
