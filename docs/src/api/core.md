# Core workflow

```@meta
CurrentModule = PEPit
```

## Main objects

```@docs
PEP
Point
Expression
Constraint
PSDMatrix
BlockPartition
PEPFunction
DualPEPCertificate
```

## Problem construction

```@docs
declare_function!
declare_block_partition!
set_initial_point!
set_initial_condition!
set_performance_metric!
add_constraint!
add_psd_matrix!
```

## Oracles and fixed points

```@docs
oracle!
gradient!
value!
stationary_point!
fixed_point!
```

## Solving and evaluation

```@docs
solve!
solve_dual!
evaluate
eval_dual
```
