# Function classes

```@meta
CurrentModule = PEPit
```

The following exported types model interpolation constraints for common
classes of scalar functions. Parameters are supplied through an
`OrderedDict`, matching the examples in `PEPit.jl/examples`.

```@docs
AbstractFunction
ConvexFunction
ConvexLipschitzFunction
SmoothFunction
SmoothConvexFunction
SmoothStronglyConvexFunction
StronglyConvexFunction
ConvexIndicatorFunction
ConvexQGFunction
ConvexSupportFunction
RsiEbFunction
SmoothConvexLipschitzFunction
SmoothStronglyConvexQuadraticFunction
SmoothQuadraticLojasiewiczFunctionCheap
SmoothQuadraticLojasiewiczFunctionExpensive
BlockSmoothConvexFunctionCheap
BlockSmoothConvexFunctionExpensive
```
