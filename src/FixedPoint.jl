module FixedPoint
function afps(f, x; iters::Int = 5000, vel::Float64 = 0.9, ep::Float64 = 0.01, tol::Float64 = 1e-12, grad_norm=x->maximum(abs,x))
    x_n = identity.(x)
    v_n = zero(x_n)
    β = vel
    ϵ = ep
    runs = 0
    for _ in 1:iters
        trial = x_n + β*v_n
        g = f(trial) - trial
        v_n = β*v_n + ϵ*g
        x_n = x_n + v_n
        runs += 1
        if grad_norm(g)<tol
            break
        end
    end
    (x = x_n, error = grad_norm(f(x_n)-x_n), iters = runs)
end
export afps
end
