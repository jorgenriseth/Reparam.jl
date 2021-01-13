using LinearAlgebra: ⋅, norm
using Parameters
using ForwardDiff: derivative

struct ReparametrizationSolution{T <: Function}
    errors::Vector{Float64}
    reparams::Vector{T} 
end

# Constructor
ReparametrizationSolution() = ReparametrizationSolution(Vector{Float64}(), Vector{Function}())


function reparametrize(q, r, projector::AbstractProjector;
        I::AbstractIntegrator=DefaultIntegrator, maxiter=30, verbosity=1, rtol=1e-3, gtol=1e-4, α_step=0.5, lsconfig=BacktrackConfig())
    # Create intial parametrization
    ψ = Diffeomorphism(projector)
    rn = ReparametrizedQmap(r, ψ)

    res = ReparametrizationSolution()
    update_result!(res, I(q, r, L2Distance())^2, ψ)

    if verbosity == 1
        printheader()
        printfirst(I(q, r, L2Distance())^2)
    end

    for i in 1:maxiter
        δE = l2_gradient(q, rn)
        ∇E = project(δE, projector)

        # Choose Step Size
        εinit  = max_step_length(∇E, alpha=α_step)
        ε = backtracking(q, rn, ∇E, εinit, I, config=lsconfig)
        #ε = εinit * 0.1
        rescale!(∇E, ε)
        γ = Reparametrization(∇E)

        update!(rn, γ)
        update_result!(res, I(q, rn, L2Distance())^2, γ)

        # Compute gradient norm and relative error to check for termination.
        δE = l2_gradient(q, rn)
        ∇E = project(δE, projector)

        gradnorm = norm(∇E.W)
        relerror = relative_error(res.errors[i+1], res.errors[i])

        if verbosity >= 1
            printline(i, I(q, rn, L2Distance())^2, εinit / α_step, ε, gradnorm, relerror)
        end

        if gradnorm < gtol || relerror < rtol
            break
        end
    end

    # Print closing line to make results prettier
    if verbosity >= 1
        printfooter()
    end
    #return res
    return res, rn
end


function reparametrize(q, r0, projector, interpolator; I::AbstractIntegrator=DefaultIntegrator,
    maxiter=50, rtol=1e-3, gtol=1e-3, α_step=0.1, lsconfig=BacktrackConfig(),
    verbosity=1) 
    id(x) = x
    ψ = id
    r = Q_reparametrization(r0, ψ)

    res = ReparametrizationSolution()
    update_result!(res, l2_distance(q, r, I=I)^2, id)

    if verbosity == 1
        printheader()
        printfirst(l2_distance(q, r, I=I)^2)
    end


    for i in 1:maxiter
        δE = l2_gradient(q, r)
        ∇E = project(δE, projector)

        # Choose Step Size
        εinit  = max_step_length(∇E, alpha=α_step)
        ε = backtracking(q, r, ∇E, εinit, config=lsconfig)
        γ(x) = x - ε * ∇E(x)
        
        φ(x) = ψ(γ(x))
        u(x) = φ(x) - x
        v = interpolate(u, interpolator)
        ψ = x -> x + v(x)

        # Update Parametrization
        r = Q_reparametrization(r0, ψ)
        update_result!(res, l2_distance(q, r, I=I)^2, γ)

        # Compute gradient norm and relative error to check for termination.
        gradnorm = norm(∇E.W)
        relerror = relative_error(res.errors[i+1], res.errors[i])

        printline(i, l2_distance(q, r, I=I)^2, εinit / α_step, ε, gradnorm, relerror)

        if gradnorm < gtol || relerror < rtol
            break
        end
    end

return res
end


function update_result!(res::ReparametrizationSolution, error, reparam)
    push!(res.errors, error)
    push!(res.reparams, reparam)
end

# struct L2Gradient{T<:Union{Qmap, ReparametrizedQmap}} <: Function
#     q::Qmap
#     r::ReparametrizedQmap
# end

function l2_gradient(q, r)
    # Get the derivatives of the curves q, r, and compute the gradient of the cost function
    qdt(t) = derivative(q, t)
    rdt(t) = derivative(r, t)
    function (t)
        return (r(t)⋅qdt(t) - q(t)⋅rdt(t))
    end
end

function relative_error(f1, f0)
    return (f0 - f1) / f0
end


function printheader(;width=82)
    println(" " * "-"^width)
    @printf "| %-4s | %-16s | %-16s | %-16s | %-16s | %-16s | \n" "Iter" "Error" "εmax" "εi" "||dγ||" "|f_i - f_{i-1}|/ |fi|"
    println("-"^width)
end


function printfirst(error)
    @printf "| %4d | %.10e | %-16s | %-16s | %-16s |\n" 0 error "" "" ""
end


function printline(iter, error, max_step, step, gradnorm, relerror)
    @printf "| %4d | %.10e | %.10e | %.10e | %.10e | %.10e |\n" iter error max_step step gradnorm relerror
end


function printfooter(;width=82)
    println(" " * "-"^width)
end


# Postprocessing
function accumulate_composition(funcvec)
    n = length(funcvec)
    out = Vector{Function}(undef, n)
    out[1] = funcvec[1]
    for i in 2:n
        out[i] = out[i-1] ∘ funcvec[i]
    end
    return out
end