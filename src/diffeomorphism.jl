struct Reparametrization{T<:BasisExpansion} <: Function
    BE::T
end

function (R::Reparametrization)(x)
    return x - R.BE(x)
end

Reparametrization(P::AbstractProjector) = Reparametrization(BasisExpansion(Float64[], eltype(P.basis)[]))

function rescale!(R::Reparametrization, a)
    rescale!(R.BE, a)
end

struct Diffeomorphism{T} <: Function
    V::Vector{T}
end

Diffeomorphism(P::AbstractProjector) = Diffeomorphism([Reparametrization(P)])

function (D::Diffeomorphism)(x)
    out = x
    for i in length(D.V):-1:1
        out = D.V[i](out)
    end
    return out
end

function update!(D::Diffeomorphism, R::Reparametrization)
    push!(D.V, R)
end