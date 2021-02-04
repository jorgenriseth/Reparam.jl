abstract type AbstractProjector end

struct OrthogonalProjector{T <: BasisFunction, T2 <: AbstractIntegrator} <: AbstractProjector
    basis::Vector{T}
    dim::Int
    I::T2
end

# Constructors
OrthogonalProjector(basis; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(basis, length(basis), I)


# Orthogonal Projection of F, using OrthogonalProjector
function project(f, P::OrthogonalProjector)
    W = [P.I(f, bi, L2InnerProduct()) for bi in P.basis]
    return BasisExpansion(W, P.basis)
end


# Interfaces for projectors
FourierVectorProjector(N; I::GaussLegendre2D=DefaultIntegrator2D) = OrthogonalProjector(FourierVectorBasis(N), 2 * (2N^2 + N), I)
FourierProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(FourierSineBasis(N), I=I)
JacobiProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(JacobiBasis(N), I=I)
PalaisProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(PalaisBasis(N), I=I)