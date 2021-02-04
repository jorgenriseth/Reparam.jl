using Parameters
using ForwardDiff: derivative, jacobian
using Printf
using LinearAlgebra: norm, det, tr

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
        @printf "Iter %4d: %12.10e vs. %12.10e\n" iter I(q, ϕ, L2Distance())^2 Inf
    end

    while I(q, ϕ, L2Distance())^2 > (E0 - ε * c * v0) && iter < maxiter
        iter += 1
        ε *= ρ
        rescale!(γ, ρ)
        if verbose
            @printf "Iter %4d: %12.10e vs. %12.10e\n" iter I(q, ϕ, L2Distance())^2 Inf
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
    vmax = -Inf
    for xi in range(0, 1, length=Nfine)
        vmax = max(vmax, derivative(v, xi))
    end
    return alpha / vmax
end


function max_step_length(v::BasisExpansion{FourierVectorBasisFunction}; Nfine=32, alpha=0.5)
    ε_bound = Inf
    A = [0. 0.; 0. 0.]
    x_alloc = [0., 0.]
    for xi in range(0, 1, length=Nfine), xj in range(0, 1, length=Nfine)
        x_alloc[1] = xi
        x_alloc[2] = xj
        A .= jacobian(v, [xi, xj])

        ε_bound = min(ε_bound, quadratic_step_selector(det(A), -tr(A)))
    end
    return ε_bound * alpha
end


function quadratic_step_selector(a, b; δ=1e-3)
    if abs(a) < δ && b < 0
        return -1 / b
    elseif (a < -δ) || (a > δ && b < -sqrt(4a))
        return -(b + sqrt(b^2 - 4a)) / (2a)
    else
        return Inf
    end
end


# Backtracking Linesearch for reparametrizatino of surfaces. 
# TODO: Generalize backtrackin to deal with both curves and surfaces.
function backtracking(q::Qmap2D, r::Union{Qmap2D, ReparametrizedQmap2D}, v::BasisExpansion, εmax, I::AbstractIntegrator; config=BacktrackConfig())
    # Initialize step length
    ε = εmax

    # Unpack Configuration
    @unpack c, ρ, maxiter, verbose = config

    # Create function stepping along v
    γ = Reparametrization(v)
    rescale!(γ, ε)
    ϕ(x) = sqrt(det(jacobian(γ, x))) * r(γ(x))

    # Get initial values for sufficient decrease condition.
    E0 = I(q, r, L2Distance())^2 
    v0 = norm(v.W)^2

    # Log Starting Values
    if verbose
        @printf "Iter %4d: %12.10e vs. %12.10e\n" iter I(q, ϕ, L2Distance())^2 Inf
    end

    # Initialize iteration
    iter = 0

    # Start loop
    while I(q, ϕ, L2Distance())^2 > (E0 - ε * c * v0) && iter < maxiter
        iter += 1
        ε *= ρ
        rescale!(γ, ρ)

        # Log Current result
        if verbose
            @printf "Iter %4d: %12.10e vs. %12.10e\n" iter I(q, ϕ, L2Distance())^2 Inf
        end
    end

    # Warn user if Backtracking not converged.
    if iter == maxiter
        println("[backtracking] Warning: Couldn't find sufficient decrease stepsize in $iter steps.")
    end

    return ε
end
