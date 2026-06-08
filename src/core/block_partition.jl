"""
    BlockPartition(d)

Represent a symbolic partition of points into `d` mutually orthogonal blocks.

Block partitions support block-coordinate interpolation constraints. Calling
[`get_block`](@ref) on a point creates symbolic block components
`x^{(1)}, ..., x^{(d)}` whose sum is the original point. The corresponding
orthogonality constraints are generated later when the PEP is compiled.

# Fields
- `d`: number of blocks.
- `list_of_constraints`: orthogonality constraints generated for this
  partition.
- `blocks_dict`: map from original points to their symbolic block components.
- `counter`: global partition index.

See also [`declare_block_partition!`](@ref), [`get_block`](@ref), and
[`get_nb_blocks`](@ref).
"""
mutable struct BlockPartition
    d::Int
    list_of_constraints::Vector{Constraint}
    blocks_dict::OrderedDict{Point,Vector{Point}}
    counter::Union{Int,Nothing}

    function BlockPartition(d::Int)
        @assert d >= 1 "BlockPartition requires a positive integer number of blocks (d >= 1)."
        bp = new(d, Constraint[], OrderedDict{Point,Vector{Point}}(), BlockPartition_counter[])
        BlockPartition_counter[] += 1
        push!(GLOBAL_BLOCK_PARTITIONS, bp)
        return bp
    end
end


"""
    get_nb_blocks(bp::BlockPartition)

Return the number of blocks in a block partition.
"""
get_nb_blocks(bp::BlockPartition) = bp.d


"""
    get_block(bp::BlockPartition, point::Point, block_number::Int)

Return the `block_number`-th symbolic block of `point`, creating the block
decomposition and its orthogonality constraints if needed.

The first `d - 1` blocks are new leaf points; the final block is defined as the
residual required to make the block sum equal to `point`. Orthogonality
constraints are registered by `add_partition_constraints!` during model
construction.
"""
function get_block(bp::BlockPartition, point::Point, block_number::Int)
    @assert 1 <= block_number <= bp.d "block_number must be an integer in 1..$(bp.d)."
    if !haskey(bp.blocks_dict, point)
        point_partition = Point[]
        accumulation = null_point
        for _ in 1:(bp.d - 1)
            new_point = Point()
            accumulation = accumulation + new_point
            push!(point_partition, new_point)
        end
        push!(point_partition, point - accumulation)
        bp.blocks_dict[point] = point_partition
    end
    return bp.blocks_dict[point][block_number]
end


add_constraint!(bp::BlockPartition, constraint::Constraint) = push!(bp.list_of_constraints, constraint)


function add_partition_constraints!(bp::BlockPartition)
    empty!(bp.list_of_constraints)
    decomposed = collect(values(bp.blocks_dict))
    for xi_decomposed in decomposed, xj_decomposed in decomposed
        for k in 1:bp.d, l in 1:(k - 1)
            add_constraint!(bp, xi_decomposed[k] * xj_decomposed[l] == 0)
        end
    end
    return bp.list_of_constraints
end
