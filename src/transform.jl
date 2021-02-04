using ForwardDiff: derivative, GradientConfig, Chunk, jacobian
using LinearAlgebra: ×, det, norm

"""
Take the q-map of a curve c(t) = [x(t), y(t)]
"""
struct Qmap{T<:Function} <: Function
    c::T
end

"""
Evaluate Qmap
"""
function (q::Qmap)(x)
    return sqrt(vecnorm(derivative(q.c, x))) * q.c(x)
end

"""
Holds a Qmap/Diffeomorphism pair, which allows iterative reparametrization.
"""
struct ReparametrizedQmap{T1 <: Qmap, T2<:Diffeomorphism} <: Function
    q::T1
    γ::T2
end

"""
Evaluate Reparametrized Qmap.
"""
function (Q::ReparametrizedQmap)(x)
    return sqrt(derivative(Q.γ, x)) * Q.q(Q.γ(x))
end

"""
2-Dimensional Qmap.
"""
struct Qmap2D{T <: Function} <: Function
    f::T
end

"""
Evaluate 2-dimensional Qmap.
"""
function (q::Qmap2D)(x)
    Df = jacobian(q.f, x)
    return sqrt(norm(Df[:, 1] × Df[:, 2])) * q.f(x)
end


"""
Function to evaluate local area scaling factor of a surface f.  Mainly helpful
for plotting.
"""
function area_scaling(f::Function)
    function (x)
        Df = jacobian(f, x)
        return norm(Df[:, 1] × Df[:, 2])
    end
end

"""
Holds a surface Qmap/Diffeomorphism pair.
"""
struct ReparametrizedQmap2D{T1 <: Qmap2D, T2 <: Diffeomorphism} <: Function
    q::T1
    γ::T2
end


"""
Evaluate Repparametrized Surface-Qmap.
"""
function (rn::ReparametrizedQmap2D)(x)
    return sqrt(det(jacobian(rn.γ, x))) * rn.q(rn.γ(x))
end

"""
Type-dependent constructor for 2Dimensional reparametrized Qmap.
Enables re-use of code for reparametrization of curves.
"""
ReparametrizedQmap(q::Qmap2D, γ) = ReparametrizedQmap2D(q, γ)


"""
Reparametrize Qmap by adding a diffeomorphism to the chain of reparametrizations 
in 
"""
function update!(Q::T, φ::Reparametrization) where T<:Union{ReparametrizedQmap, ReparametrizedQmap2D}
    update!(Q.γ, φ)
end
