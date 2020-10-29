using FastGaussQuadrature: gausslegendre
using LinearAlgebra: ⋅

abstract type AbstractIntegrator end

struct GaussLegendre <: AbstractIntegrator
    N::Int
    weights::Vector{Float64}
    nodes::Vector{Float64}
end


# Constsructor
function GaussLegendre(N)
    nodes, weights = gausslegendre(N)
    nodes .= 0.5 * nodes .+ 0.5 # Translate nodes from [-1, 1] to [0, 1]
    return GaussLegendre(N, weights, nodes)
end


# Default Instance
DefaultIntegrator = GaussLegendre(1000)


function integrate(f, I::AbstractIntegrator)
    out = 0.0
    for i in 1:I.N
        out += f(I.nodes[i]) * I.weights[i]
    end
    return out * 0.5
end