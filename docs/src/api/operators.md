# Operator classes

```@meta
CurrentModule = PEPit
```

Operator classes use the same oracle infrastructure as function classes, but
their interpolation constraints describe set-valued or single-valued operators
such as monotone, cocoercive, Lipschitz, or nonexpansive maps.

```@docs
LipschitzOperator
LinearOperator
NonexpansiveOperator
MonotoneOperator
StronglyMonotoneOperator
NegativelyComonotoneOperator
CocoerciveOperator
CocoerciveStronglyMonotoneOperatorCheap
CocoerciveStronglyMonotoneOperatorExpensive
LipschitzStronglyMonotoneOperatorCheap
LipschitzStronglyMonotoneOperatorExpensive
SymmetricLinearOperator
SkewSymmetricLinearOperator
```
