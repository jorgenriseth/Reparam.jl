struct GaussianRadialBasisFunction <: BasisFunction
    center::Float64
    param::Float64
end


# Callable
function (g::GaussianRadialBasisFunction)(x)
    return exp(-(g.param * (x - g.center))^2)
end


struct Interpolator{T<:BasisFunction}
    basis::Vector{T}  # Interpolation basis functions.
    M::Matrix{Float64}  # Mass matrix Mij = <bi, bj> .
    b::Vector{Float64}  # Pre-allocated interpolation vector.
    w::Vector{Float64} # Pre-allocated solution vector.
    nodes::Vector{Float64}  # Interpolation nodes.
    dim::Int  # Number of basis functions.
end

# Constructor
function Interpolator(basis::Vector{T}) where {T<:BasisFunction}
    n = length(basis)
    M = [bi(bj.center) for bj in basis, bi in basis]
    nodes = [bi.center for bi in basis]
    return Interpolator(basis, M, zeros(n), zeros(n), nodes, n)
end


function GaussianRadialBasis(N)
    param = (N-1)
    return [GaussianRadialBasisFunction(xi, param) for xi in range(0, 1, length=N)]
end


function interpolate(f, IP::Interpolator)
    IP.b .= f.(IP.nodes)
    IP.w .= IP.M \ IP.b
    return BasisExpansion(IP.w, IP.basis)
end


RBFInterpolator(N) = Interpolator(GaussianRadialBasis(N))