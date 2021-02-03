abstract type AbstractProjector end

struct OrthogonalProjector{T <: BasisFunction, T2 <: AbstractIntegrator} <: AbstractProjector
    basis::Vector{T}
    dim::Int
    I::T2
end

# Constructors
OrthogonalProjector(basis; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(basis, length(basis), I)

# Interfaces for projectors
FourierProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(FourierSineBasis(N), I=I)
JacobiProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(JacobiBasis(N), I=I)
PalaisOrthogonalProjector(N; I::AbstractIntegrator=DefaultIntegrator) = OrthogonalProjector(PalaisBasis(N), I=I)

function project(f, P::OrthogonalProjector)
    W = [P.I(f, bi, L2InnerProduct()) for bi in P.basis]
    return BasisExpansion(W, P.basis)
end


struct PalaisProjector <: AbstractProjector
    basis::Vector{PalaisBasisFunction}
    dim::Int
    I::GaussLegendre
end

PalaisProjector(N, I::GaussLegendre) = PalaisProjector(PalaisBasis(N), N, I)

function project(f, P::PalaisProjector)
    W = [P.I(f, bi, PalaisInnerProduct()) for bi in P.basis]
    return BasisExpansion(W, P.basis)
end


FourierVectorProjector(N; I::GaussLegendre2D=DefaultIntegrator2D) = OrthogonalProjector(FourierVectorBasis(N), 2 * (2N^2 + N), I)