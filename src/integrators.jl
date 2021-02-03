using FastGaussQuadrature: gausslegendre
using LinearAlgebra: ⋅
using ForwardDiff: derivative

abstract type AbstractIntegrator end
abstract type Operator end
struct L2Distance end
struct L2InnerProduct end
struct L2Norm end
struct PalaisInnerProduct end 

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

function squarenorm(x::AbstractVector)
    return x[1]^2 + x[2]^2
end 

function squarenorm(x)
    return x^2
end 

function vecnorm(x::AbstractVector)
    return sqrt(x[1]^2 + x[2]^2)
end 

function vecnorm(x)
    return x
end 

function vecdot(x::AbstractVector, y::AbstractVector)
    return  x[1] * y[1] + x[2] * y[2]
end

function vecdot(x, y)
    return  x * y 
end


# Default Instance
DefaultIntegrator = GaussLegendre(128)


function integrate(f, I::AbstractIntegrator)
    out = 0.0
    for i in 1:I.N
        out += f(I.nodes[i]) * I.weights[i]
    end
    return out * 0.5
end


function (I::GaussLegendre)(f)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N
        out += f(I.nodes[i]) * I.weights[i]
    end
    return out * 0.5
end

function (I::GaussLegendre)(f, F::L2Norm)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N
        out += squarenorm(f(I.nodes[i])) * I.weights[i]
    end
    return sqrt(out * 0.5)
end

function (I::GaussLegendre)(f, g, F::L2Distance)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N
        out += squarenorm(f(I.nodes[i]) - g(I.nodes[i])) * I.weights[i]
    end
    return sqrt(out * 0.5)
end

function (I::GaussLegendre)(f, g, F::L2InnerProduct)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N
        out += vecdot(f(I.nodes[i]),  g(I.nodes[i])) * I.weights[i]
    end
    return out * 0.5
end

function (I::GaussLegendre)(f, g, F::PalaisInnerProduct)
    df(x) = derivative(f, x)
    dg(x) = derivative(g, x)
    out = 0.0
    
    @inbounds @simd for i in 1:I.N
        out += vecdot(df(I.nodes[i]),  dg(I.nodes[i])) * I.weights[i]
    end
    return out * 0.5
end