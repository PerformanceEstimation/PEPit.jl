# Primitive steps

```@meta
CurrentModule = PEPit
```

Primitive steps create symbolic points, gradients, function values, and
constraints for common algorithmic operations. They are intended to keep example
scripts close to the mathematical method being analyzed.

```@docs
inexact_gradient_step!
bregman_gradient_step!
bregman_proximal_step!
epsilon_subgradient_step!
exact_linesearch_step!
inexact_proximal_step!
proximal_step!
linear_optimization_step!
shifted_optimization_step!
```
