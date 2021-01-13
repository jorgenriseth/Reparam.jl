using Parameters
using ForwardDiff: derivative
using Printf
using LinearAlgebra: norm

@with_kw struct BacktrackConfig
    c::Float64 = 0.1 # Weak wolfe constant
    ρ::Float64 = 0.9 # Step size reduction constant
    maxiter::Int = 100 # Maximum backtracking iterations
    verbose::Bool = false # Should backtrack give output?
end


""" Algorithm 3.1. Nocedal & Wright Numerical Optimization: Backtracking Line Search
Simple approach for choosing the step size in a line serach optimization algorithm. The algorithm
either takes a selected step lengtho f alphamax, or a short enough step to satisfy sufficient decrease
condition with constant c."""
function backtracking(q::Function, r::Function, v::BasisExpansion, εmax, I::AbstractIntegrator; config=BacktrackConfig())
    # Initialize step length
    ε = εmax

    # Unpack Configuration
    @unpack c, ρ, maxiter, verbose = config

    # Create function stepping along v
    γ = Reparametrization(v)
    rescale!(γ, ε)
    ϕ(x) = sqrt(derivative(γ, x)) * r(γ(x))

    # Get initial values for sufficient decrease condition.
    E0 = I(q, r, L2Distance())^2 
    v0 = norm(v.W)^2

    # Initialize iteration and
    iter = 0
    if verbose
        @printf "Iter %4d: %12.10e vs. %12.10e\n" iter l2_distance(q, ϕ, I=I)^2 l(ε)
    end
    while I(q, ϕ, L2Distance())^2 > (E0 - ε * c * v0) && iter < maxiter
        iter += 1
        ε *= ρ
        rescale!(γ, ρ)
        if verbose
            @printf "Iter %4d: %12.10e vs. %12.10e\n" iter l2_distance(q, ϕ, I=I)^2 l(ε)
        end
    end

    # Warn user if backtracking didn't work
    if iter == maxiter
        println("[backtracking] Warning: Couldn't find sufficient decrease stepsize in $iter steps.")
    end

    return ε
end


function make_weak_wolfe_line(E0, v_norm, c)
    a = c * v_norm
    function (ε)
        return E0 - a * ε
    end
end


# function max_step_length(v; Nfine=201, alpha=0.99)
#     vdt(x) = derivative(v, x)
#     vi = vdt.(range(0, 1, length=Nfine))
#     return alpha / maximum(vi)
# end

function max_step_length(v; Nfine=201, alpha=0.99)
    vmax = -Inf
    for xi in range(0, 1, length=Nfine)
        vmax = max(vmax, derivative(v, xi))
    end
    return alpha / vmax
end