using Reparam
using Test

@testset "Interpolation" begin
    include("test_lagrange.jl")
    include("test_rbf.jl")
end

@testset "Gradient Descent" begin
    include("test_gradient_descent.jl")
end
