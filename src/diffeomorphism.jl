"""
Single diffeomorphism given as a small perturbation of the identity. Takes
 a BasisExpansion-type, but evaluates by
    γ(x) = x + BE(x)
"""
struct Reparametrization{T<:BasisExpansion} <: Function
    BE::T
end


# Make Type Callable
function (R::Reparametrization)(x)
    return x - R.BE(x)
end

# Constructor for empty Reparametrization
Reparametrization(P::AbstractProjector) = Reparametrization(BasisExpansion(Float64[], eltype(P.basis)[]))

function rescale!(R::Reparametrization, a)
    rescale!(R.BE, a)
end


# Diffeomorphism, defined as chain of Reparametrizations
struct Diffeomorphism{T} <: Function
    V::Vector{T}
end


# Constructor for identity-diffeomorphism
Diffeomorphism(P::AbstractProjector) = Diffeomorphism([Reparametrization(P)])


# Make Diffeomorphism callable
function (D::Diffeomorphism)(x)
    out = x
    for i in length(D.V):-1:1
        out = D.V[i](out)
    end
    return out
end


# Add a reparametrization to the chain of diffeomorphisms
function update!(D::Diffeomorphism, R::Reparametrization)
    push!(D.V, R)
end