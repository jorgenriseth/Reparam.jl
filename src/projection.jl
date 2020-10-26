abstract type AbstractProjector end 

struct OrthogonalProjector{T<:BasisFunction, F<:Function} <:AbstractProjector
    basis::Vector{T}
    inner_product::F
    dim::Int
end

struct InterpolationProjector{T<:BasisFunction, TF<:Real} <:AbstractProjector
    basis::Vector{T}
    mass_matrix::Matrix{TF}
    dim::Int
    
    function InterpolationProjector(basis::Vector{T}) where {T<:BasisFunction}
        A = [bi(bj.center) for bj in basis, bi in basis]
        new{T, typeof(A[1, 1])}(basis, A, length(basis))
    end
end


function project(f, OP::OrthogonalProjector)
    W = [OP.inner_product(f, bi) for bi in OP.basis]
    return BasisExpansion(W, OP.basis)
end

function project(f, OP::OrthogonalProjector, N::Int)
    W = [OP.inner_product(f, bi) for bi in OP.basis[1:N]]
    return BasisExpansion(W, OP.basis[1:N])
end


function project(f, IP::InterpolationProjector)
    b = load_vector(f, [bi.center for bi in IP.basis])
    W = IP.mass_matrix \ b
    return BasisExpansion(W, IP.basis)
end


""" Create the interpolation load vector, which should be the function values on internal nodes, and zero
on the boundaries."""
function load_vector(f, nodes)
    b = zeros(length(nodes))
    b[2:end-1] = [f(xi) for xi in nodes[2:end-1]]
    return b
end


FourierSineProjector(N::Int) = OrthogonalProjector(FourierSineBasis(N), l2_inner_product, N)
PalaisProjector(N::Int) = error("Not Implemented Properly.") # OrthogonalProjector(PalaisBasis(N), palais_inner_product, N)
JacobiProjector(N::Int) = OrthogonalProjector(JacobiBasis(N), l2_inner_product, N)

function GaussianInterpolationProjector(N::Int; param=nothing)
    @assert N > 2 "Need at least 3 basis function for interpolation."
    basis = GaussianRadialBasis(N, param=param)
    return InterpolationProjector(basis)
end