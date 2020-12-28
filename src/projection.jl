abstract type AbstractProjector end

struct OrthogonalProjector{T <: BasisFunction} <: AbstractProjector
    basis::Vector{T}
    dim::Int
    I::GaussLegendre
end

# Constructors
OrthogonalProjector(basis; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(basis, length(basis), I)

# Interfaces for projectors
FourierProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(FourierSineBasis(N), I=I)
JacobiProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(JacobiBasis(N), I=I)


function project(f, P::OrthogonalProjector)
    W = [l2_inner_product(f, bi; I=P.I) for bi in P.basis]
    return BasisExpansion(W, P.basis)
end