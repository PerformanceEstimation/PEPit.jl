@doc raw"""
    BlockSmoothConvexFunctionExpensive(param; reuse_gradient=true)

Class of convex functions that are smooth by blocks (with one smoothness
parameter ``L_k`` per block of a [`BlockPartition`](@ref)), modeled through the
strengthened necessary interpolation constraints of [2, Section 3.1]. Compared
with [`BlockSmoothConvexFunctionCheap`](@ref), the constraints are tighter but
significantly more expensive (they involve one PSD block per triplet of oracle
points and pair of blocks).

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
denoting by ``v^{(m)}`` the block-``m`` component of a point ``v``, the
following holds for every triplet of oracle points ``(i, j, k)`` and every pair
of blocks ``(m, l)``: defining the block-``m`` smooth-convex surpluses

```math
\begin{aligned}
A & = f_i - f_k - \langle g_k, x_i - x_k \rangle - \frac{1}{2 L_m} \|g_i^{(m)} - g_k^{(m)}\|^2, \\
B & = f_i - f_j - \langle g_j, x_i - x_j \rangle - \frac{1}{2 L_m} \|g_i^{(m)} - g_j^{(m)}\|^2, \\
C & = \frac{1}{2 L_m} \|g_j^{(m)} - g_k^{(m)}\|^2 - \frac{1}{2 L_l} \|g_j^{(l)} - g_k^{(l)}\|^2,
\end{aligned}
```

there exists a scalar slack ``\lambda \geqslant 0`` such that

```math
\begin{pmatrix} A & \tfrac{1}{2}(A + B + C) - \lambda \\
\tfrac{1}{2}(A + B + C) - \lambda & B \end{pmatrix} \succeq 0.
```

The slack ``\lambda`` is modeled by a fresh [`Expression`](@ref) and the
``2 \times 2`` condition by a [`PSDMatrix`](@ref); see [2, Section 3.1].

# Julia usage
```julia
problem = PEP()
partition = declare_block_partition!(problem, 3)
param = OrderedDict("partition" => partition, "L" => [1.0, 4.0, 10.0])
f = declare_function!(problem, BlockSmoothConvexFunctionExpensive, param)
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

[[2] A. Rubbens, J.M. Hendrickx, A. Taylor (2025).
A constructive approach to strengthen algebraic descriptions of function and
operator classes.](https://arxiv.org/pdf/2504.14377.pdf)

See also [`declare_function!`](@ref), [`declare_block_partition!`](@ref), and
[`BlockSmoothConvexFunctionCheap`](@ref).
"""
mutable struct BlockSmoothConvexFunctionExpensive <: AbstractFunction
    partition::BlockPartition
    L::Vector{Float64}
    _PEPit_func::PEPFunction

    function BlockSmoothConvexFunctionExpensive(param; is_leaf=true, decomposition_dict=nothing, reuse_gradient=true)
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

gradient!(f::BlockSmoothConvexFunctionExpensive, p::Point) = gradient!(f._PEPit_func, p)
value!(f::BlockSmoothConvexFunctionExpensive, p::Point) = value!(f._PEPit_func, p)
stationary_point!(f::BlockSmoothConvexFunctionExpensive) = stationary_point!(f._PEPit_func)
add_constraint!(func::BlockSmoothConvexFunctionExpensive, constraint::Constraint) = add_constraint!(func._PEPit_func, constraint)

function add_class_constraints!(func::BlockSmoothConvexFunctionExpensive)
    internal = func._PEPit_func
    points = internal.list_of_points
    nb = get_nb_blocks(func.partition)
    n = length(points)
    for i in 1:n, j in 1:n, k in 1:n
        (i == j && i == k) && continue
        xi, gi, fi = points[i]
        xj, gj, fj = points[j]
        xk, gk, fk = points[k]
        for m in 1:nb, l in 1:nb
            gim = get_block(func.partition, gi, m)
            gjm = get_block(func.partition, gj, m)
            gkm = get_block(func.partition, gk, m)
            gjl = get_block(func.partition, gj, l)
            gkl = get_block(func.partition, gk, l)

            new_expr = Expression()
            add_constraint!(func, new_expr >= 0)

            A = -(-fi + fk + gk * (xi - xk) + 1 / (2 * func.L[m]) * (gim - gkm)^2)
            B = -(-fi + fj + gj * (xi - xj) + 1 / (2 * func.L[m]) * (gim - gjm)^2)
            C = -(1 / (2 * func.L[l]) * (gjl - gkl)^2 - 1 / (2 * func.L[m]) * (gjm - gkm)^2)

            off = 0.5 * (A + B + C) - new_expr
            T = Matrix{Expression}(undef, 2, 2)
            T[1, 1] = A; T[1, 2] = off; T[2, 1] = off; T[2, 2] = B
            push!(internal.list_of_class_psd, PSDMatrix(matrix_of_expressions=T))
        end
    end
end

_get_pep_func(f::BlockSmoothConvexFunctionExpensive) = f._PEPit_func
