using Parameters
using ForwardDiff: derivative
using Printf

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
function backtracking(q, r, v::BasisExpansion, εmax; I::AbstractIntegrator=DefaultIntegrator, config=BacktrackConfig())
    # Initialize step length
    ε = εmax

    # Unpack Configuration
    @unpack c, ρ, maxiter, verbose = config

    # Create function stepping along v
    γ(x) = x - ε * v(x)
    ϕ = Q_reparametrization(r, γ)


    # Get initial values for sufficient decrease condition.
    E0 = l2_distance(q, r, I=I)^2
    v0 = l2_norm(v, I=I)^2

    l = make_weak_wolfe_line(E0, v0, c)

    # Initialize iteration and
    iter = 0
    while l2_distance(q, ϕ, I=I)^2 > l(ε) && iter < maxiter
        iter += 1
        ε *= ρ
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


function max_step_length(v; Nfine=201, alpha=0.99)
    vdt(x) = derivative(v, x)
    vi = vdt.(range(0, 1, length=Nfine))
    return alpha / maximum(vi)
end