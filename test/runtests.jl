using Reparam
using Test

@testset "Interpolation" begin
    include("test_interpolation.jl")
end

@testset "Gradient Descent" begin
    include("test_gradient_descent.jl")
end
