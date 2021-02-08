using FastGaussQuadrature: gausslegendre
using LinearAlgebra: ⋅, norm
using ForwardDiff

# Create simple 2-Dimensional Gauss-Legendre Quadrature struct
struct GaussLegendre2D <: AbstractIntegrator
    N::Int
    weights::Vector{Float64}
    nodes::Vector{Vector{Float64}}
end


# Constructor for N-point GaussLegendre 2-dimensional quadrature
function GaussLegendre2D(N)
    nodes, weights = gausslegendre(N)
    nodes .= 0.5 * nodes .+ 0.5 # Translate nodes from [-1, 1] to [0, 1]

    nodes2D = [[xi, xj] for xi in nodes for xj in nodes]
    weights2D = [wi * wj for wi in weights for wj in weights]
    return GaussLegendre2D(N, weights2D, nodes2D)
end


# Default Instance with 32×32=1024 quadrature points
DefaultIntegrator2D = GaussLegendre2D(32)


# Integrate a function f using 2d GL-quadrature
function (I::GaussLegendre2D)(f)
    out = 0.0
    @inbounds @simd for i in 1:I.N^2
        out += f(I.nodes[i]) * I.weights[i]
    end
    return out * 0.25
end


# Compute the L2-norm of f using 2D GL-quadradture
function (I::GaussLegendre2D)(f, F::L2Norm)
    out = 0.0
    @inbounds @simd for i in 1:I.N^2
        out += norm(f(I.nodes[i]))^2 * I.weights[i]
    end
    return sqrt(out * 0.25)
end


# Compute teh L2-distance between f and g using 2D GL-quadrature
function (I::GaussLegendre2D)(f, g, F::L2Distance)
    out = 0.0
    @inbounds @simd for i in 1:I.N^2
        out += norm(f(I.nodes[i]) - g(I.nodes[i]))^2 * I.weights[i]
    end
    return sqrt(out * 0.25)
end


# Compute the L2 inner-product between function f and g using 2D GL-quadrature
function (I::GaussLegendre2D)(f, g, F::L2InnerProduct)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N^2
        out += (f(I.nodes[i]) ⋅ g(I.nodes[i]) )* I.weights[i]
    end
    return out * 0.25
end