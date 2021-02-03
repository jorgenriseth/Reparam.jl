using FastGaussQuadrature: gausslegendre
using LinearAlgebra: ⋅, norm
using ForwardDiff

struct GaussLegendre2D <: AbstractIntegrator
    N::Int
    weights::Vector{Float64}
    nodes::Vector{Vector{Float64}}
end

function GaussLegendre2D(N)
    nodes, weights = gausslegendre(N)
    nodes .= 0.5 * nodes .+ 0.5 # Translate nodes from [-1, 1] to [0, 1]

    nodes2D = [[xi, xj] for xi in nodes for xj in nodes]
    weights2D = [wi * wj for wi in weights for wj in weights]
    return GaussLegendre2D(N, weights2D, nodes2D)
end

# Default Instance
DefaultIntegrator2D = GaussLegendre2D(16)


function (I::GaussLegendre2D)(f)
    out = 0.0
    @inbounds @simd for i in 1:I.N^2
        out += f(I.nodes[i]) * I.weights[i]
    end
    return out * 0.25
end

function (I::GaussLegendre2D)(f, F::L2Norm)
    out = 0.0
    @inbounds @simd for i in 1:I.N^2
        out += norm(f(I.nodes[i]))^2 * I.weights[i]
    end
    return sqrt(out * 0.25)
end

function (I::GaussLegendre2D)(f, g, F::L2Distance)
    out = 0.0
    @inbounds @simd for i in 1:I.N^2
        out += norm(f(I.nodes[i]) - g(I.nodes[i]))^2 * I.weights[i]
    end
    return sqrt(out * 0.25)
end

function (I::GaussLegendre2D)(f, g, F::L2InnerProduct)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N^2
        out += (f(I.nodes[i]) ⋅ g(I.nodes[i]) )* I.weights[i]
    end
    return out * 0.25
end

function (I::GaussLegendre2D)(f, g, F::PalaisInnerProduct)
    error("NotImplementedError")
    df(x) = derivative(f, x)
    dg(x) = derivative(g, x)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N^2
        out += vecdot(df(I.nodes[i]),  dg(I.nodes[i])) * I.weights[i]
    end
    return out * 0.5
end