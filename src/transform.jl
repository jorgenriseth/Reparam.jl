using ForwardDiff: derivative, GradientConfig, Chunk
using LinearAlgebra: norm

# Q-transform and functions 
struct Qmap{T<:Function} <: Function
    c::T
end

#Qmap(c::Function) = Qmap(c)

function (q::Qmap)(x)
    return sqrt(vecnorm(derivative(q.c, x))) * q.c(x)
end

# function Qmap(c::Function, dc::Function)
#     function (x)
#         return sqrt(derivative(x, c)) * c(x)
#     end
# end

# Add qmap-diffeomorphism pair
struct ReparametrizedQmap{T1 <: Qmap, T2<:Diffeomorphism} <: Function
    q::T1
    γ::T2
end

ReparametrizedQmap(q, γ) = ReparametrizedQmap(q, γ)

function (Q::ReparametrizedQmap)(x)
    return sqrt(derivative(Q.γ, x)) * Q.q(Q.γ(x))
end

function update!(Q::ReparametrizedQmap, φ::Reparametrization)
    update!(Q.γ, φ)
end

function Q_transform(c)
    cdt(t) = derivative(c, t)
    function (t)
        return sqrt(norm(cdt(t))) * c(t)
    end
end


function Q_reparametrization(q, γ)
    γdt(t) = derivative(γ, t)
    function (t)
        return √(norm(γdt(t))) * q(γ(t))
    end
end


struct FiniteDifference; h::Float64; end

function Q_transform(c, FD::FiniteDifference)
    function (x)
        return sqrt(norm((c(x+FD.h) - c(x-FD.h))/(2*FD.h))) * c(x)
    end
end


function Q_reparametrization(q, γ, FD::FiniteDifference)
    function (x)
        return √norm((γ(x+FD.h) - γ(x-FD.h))/(2* FD.h)) * q(γ(x))
    end
end