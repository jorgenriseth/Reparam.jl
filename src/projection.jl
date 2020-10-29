abstract type AbstractProjector end

struct OrthogonalProjector{T <: BasisFunction} <: AbstractProjector
    basis::Vector{T}
    dim::Int
    I::GaussLegendre
end

# Constructors
OrthogonalProjector(basis) = OrthogonalProjector(basis, length(basis), DefaultIntegrator)
OrthogonalProjector(basis, I) = OrthogonalProjector(basis, length(basis), I)


# Interfaces for projectors
FourierProjector(N) = OrthogonalProjector(FourierSineBasis(N))
JacobiProjector(N) = OrthogonalProjector(JacobiBasis(N))

function project(f, P::OrthogonalProjector)
    W = [l2_inner_product(f, bi; I=P.I) for bi in P.basis]
    return BasisExpansion(W, P.basis)
end