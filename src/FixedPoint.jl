"""
    FixedPoint.jl
exports `afps` function, for further help type `?afps` in the REPL.
"""
module FixedPoint
"""
    afps(f, x; iters::Int = 5000, vel::Float64 = 0.9, ep::Float64 = 0.01, tol::Float64 = 1e-12, grad_norm=x->maximum(abs,x))

solve equation `f(x) = x` according to:

    `f` : function to find fixed point for

    `x` : initial condition, ideally it should be close to the final solution

    `vel` : amount of Nesterov acceleration in [0,1]

    `ep` : learning rate, typically in ]0,1[

    `tol` : absolute tolerance on |f(x)-x|

    `grad_norm` : function to evaluate the norm for |f(x)-x|

returns a named tuple (x, error, iters) where:

    `x` : is the solution found for f(x)=x

    `error` : is the norm of f(x)-x at the solution point

    `iters` : total number of iterations performed
"""
function afps(
    f::Fun,
    x::Mat;
    iters::Int = 5000,
    vel::T = 0.9,
    ep::T = 0.01,
    tol::T = 1e-12,
    grad_norm = x -> maximum(abs, x),
) where {T <: Number, Mat<:Union{AbstractArray{T},T}, Fun <: Function}
    x_n = identity.(x)
    v_n = zero(x_n)
    β = vel
    ϵ = ep
    runs = 0
    for _ = 1:iters
        trial = x_n + β * v_n
        g = f(trial) - trial
        v_n = β * v_n + ϵ * g
        x_n = x_n + v_n
        runs += 1
        if grad_norm(g) < tol
            break
        end
    end
    (x = x_n, error = grad_norm(f(x_n) - x_n), iters = runs)
end


"""
    afps!(f!, x; iters::Int = 5000, vel::Float64 = 0.9, ep::Float64 = 0.01, tol::Float64 = 1e-12, grad_norm=x->maximum(abs,x))

solve equation `f(x) = x` according to:

    `f!` : inplace version of function to find fixed point for, calling `f!(out,x)` should amount to writing `out = f(x)`

    `x` : initial condition, ideally it should be close to the final solution

    `vel` : amount of Nesterov acceleration in [0,1]

    `ep` : learning rate, typically in ]0,1[

    `tol` : absolute tolerance on |f(x)-x|

    `grad_norm` : function to evaluate the norm for |f(x)-x|

returns a named tuple (x, error, iters) where:

    `x` : is the solution found for f(x)=x

    `error` : is the norm of f(x)-x at the solution point

    `iters` : total number of iterations performed
"""
function afps!(
    f!::Fun,
    x_n::Mat;
    iters::Int = 5000,
    vel::T = 0.9,
    ep::T = 0.01,
    tol::T = 1e-12,
    grad_norm = x -> maximum(abs, x),
) where {T <: Number, Mat<:Union{AbstractArray{T},T}, Fun <: Function}
    v_n = zero(x_n)
    trial = zero(x_n)
    g = zero(x_n)
    β = vel
    ϵ = ep
    runs = 0
    for _ = 1:iters
        #prediction using velocity
        x_n .+= β .* v_n
        #eval gradient
        f!(g,x_n)
        g .-= x_n
        #correction
        x_n .+= ϵ .* g
        #update velocity 
        v_n .= β .* v_n .+ ϵ .* g
        runs += 1
        if grad_norm(g) < tol
            break
        end
    end
    (x = x_n, error = grad_norm(g), iters = runs)
end


export afps,afps!
end
