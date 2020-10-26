using FastGaussQuadrature: gausslegendre

abstract type Integrator end

""" GaussLegendre Integration with fixed number of quadrature nodes and weights."""
struct GaussLegendre{T1<:Number, T2<:Integer} <: Integrator
    interval::Tuple{T1, T1}
    npoints::T2
    weights::Vector{Float64}
    nodes::Vector{Float64}
end


function GaussLegendre(interval, npoints)
    a, b = interval
    nodes, weights = gausslegendre(npoints)
    nodes .= (b - a) / 2. * nodes .+ (b + a)/ 2.
    return GaussLegendre(interval, npoints, weights, nodes)
end



function (I::GaussLegendre)(f::Function)
    out::Float64 = 0.0
    for i in 1:I.npoints
        out += f(I.nodes[i]) * I.weights[i]
    end
    out * (I.interval[2] - I.interval[1]) / 2.
end


DefaultIntegrator = GaussLegendre((0, 1), 1000)