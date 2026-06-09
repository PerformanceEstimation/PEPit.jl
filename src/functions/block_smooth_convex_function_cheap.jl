@doc raw"""
    BlockSmoothConvexFunctionCheap(param; reuse_gradient=true)

Class of convex functions that are smooth by blocks (with one smoothness
parameter ``L_k`` per block of a [`BlockPartition`](@ref)), modeled through a
cheap set of necessary interpolation constraints.

Overrides `add_class_constraints!` to add the conditions of the class when
[`solve!`](@ref) builds the SDP.

!!! warning
    Functions that are smooth by blocks and convex generally do not enjoy known
    interpolation conditions. The conditions implemented in this class are
    necessary but a priori not sufficient for interpolation. Hence, the
    numerical results obtained when using this class might be non-tight upper
    bounds.

# Class parameters
- `param["partition"]`: the [`BlockPartition`](@ref) of the variables.
- `param["L"]`: smoothness parameters (a vector with one entry per block, or a scalar for a single block).

# Necessary conditions
Associating with each oracle call ``i`` the triplet ``(x_i, g_i, f_i)`` and
denoting by ``v^{(k)}`` the block-``k`` component of a point ``v`` (see
[`get_block`](@ref)), the following constraint is added for every pair
``i \neq j`` and every block ``k`` (see [1]):

```math
f_i - f_j \geqslant \langle g_j, x_i - x_j \rangle
+ \frac{1}{2 L_k} \left\| g_i^{(k)} - g_j^{(k)} \right\|^2.
```

# Julia usage
```julia
problem = PEP()
partition = declare_block_partition!(problem, 3)
param = OrderedDict("partition" => partition, "L" => [1.0, 4.0, 10.0])
f = declare_function!(problem, BlockSmoothConvexFunctionCheap, param)
```

!!! note
    Smooth convex functions by blocks are necessarily differentiable, hence
    `reuse_gradient` is set to `true`.

# Fields
- `partition::BlockPartition`: partitioning of the variables.
- `L::Vector{Float64}`: smoothness parameters, one per block.
- `_PEPit_func`: internal [`PEPFunction`](@ref) storing oracle calls and constraints.

# References

[[1] Z. Shi, R. Liu (2016).
Better worst-case complexity analysis of the block coordinate descent method
for large scale machine learning. In 2017 16th IEEE International Conference on
Machine Learning and Applications (ICMLA).](https://arxiv.org/pdf/1608.04826.pdf)

See also [`declare_function!`](@ref), [`declare_block_partition!`](@ref), and
[`BlockSmoothConvexFunctionExpensive`](@ref).
"""
mutable struct BlockSmoothConvexFunctionCheap <: AbstractFunction
    partition::BlockPartition
    L::Vector{Float64}
    _PEPit_func::PEPFunction

    function BlockSmoothConvexFunctionCheap(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
        @assert is_leaf
        func = PEPFunction(is_leaf=is_leaf, decomposition_dict=decomposition_dict, reuse_gradient=reuse_gradient)
        partition = param["partition"]
        Lraw = param["L"]
        L = Lraw isa AbstractVector ? Float64.(Lraw) : [Float64(Lraw)]
        if get_nb_blocks(partition) > 1
            @assert length(L) == get_nb_blocks(partition) "length(L) must equal the number of blocks."
            @assert all(isfinite, L)
        end
        return new(partition, L, func)
    end
end

gradient!(f::BlockSmoothConvexFunctionCheap, p::Point) = gradient!(f._PEPit_func, p)
value!(f::BlockSmoothConvexFunctionCheap, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::BlockSmoothConvexFunctionCheap) = stationary_point!(f._PEPit_func)
add_constraint!(func::BlockSmoothConvexFunctionCheap, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::BlockSmoothConvexFunctionCheap)
    points = func._PEPit_func.list_of_points
    nb = get_nb_blocks(func.partition)
    for (i, (xi, gi, fi)) in enumerate(points), (j, (xj, gj, fj)) in enumerate(points)
        i == j && continue
        for k in 1:nb
            gik = get_block(func.partition, gi, k)
            gjk = get_block(func.partition, gj, k)
            add_constraint!(func, fi - fj >= gj * (xi - xj) + 1 / (2 * func.L[k]) * (gik - gjk)^2)
        end
    end
end

_get_pep_func(f::BlockSmoothConvexFunctionCheap) = f._PEPit_func
