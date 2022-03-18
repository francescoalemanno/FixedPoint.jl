using FixedPoint, Test
@testset "Vector Equation" begin
    Ts = LinRange(0.01,2.0,500)
    βs = 1 ./ Ts
    mag = afps(x->tanh.(βs.*x), zero(βs).+1, grad_norm=x->maximum(abs,x), iters=5000)
    @test mag.error<1e-4
end