# FixedPoint.jl

    afps(f, x; iters::Int = 5000, vel::Float64 = 0.9, ep::Float64 = 0.01, tol::Float64 = 1e-12, grad_norm=x->maximum(abs,x))

solve equation `f(x) = x` according to:

    `f` : function to find fixed point for

    `x` : initial condition, ideally it should be close to the final solution

    `vel` : amount of Nesterov acceleration in [0,1]

    `ep` : learning rate, typically in ]0,1[

    `tol` : absolute tolerance on |f(x)-x|

    `grad_norm` : function to evaluate the norm |f(x)-x|

returns a named tuple (x, error, iters) where:

    `x` : is the solution found for f(x)=x

    `error` : is the norm of f(x)-x at the solution point

    `iters` : total number of iterations performed


# Examples
## Scalar function example

```julia
using FixedPoint
s = afps(x -> 2 - x^2 + x, 1.3)
@show s.x, √2
```

## Vector function example
```julia
using FixedPoint, LinearAlgebra
Ts = LinRange(0.01, 2.0, 500)
βs = 1 ./ Ts
f(x) = tanh.(βs .* x)
s = afps(
    f,
    zero(βs) .+ 1,
    iters = 5000,
)
@show norm(f(s.x).-s.x)
```

## Inplace version
for the inplace method use `afps!` as
```julia
Ts = LinRange(0.01, 2.0, 500)
βs = 1 ./ Ts
function f!(out,x)
    @. out = tanh(βs * x)
end
x  = zero(βs) .+ 1 
afps!(
    f!,
    x,
    grad_norm = x -> maximum(abs, x),
    iters = 5000,
)
@show maximum(abs, x .- tanh.(βs .* x))
```