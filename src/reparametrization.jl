using LinearAlgebra: ⋅


function reparametrize(q, r, projector::AbstractProjector;
    maxiter=20, verbosity=0, rtol=1e-3, gtol=1e-4)
    @assert maxiter > 0

    # Create intial parametrization
    id(x) = x


    # Create Result-container.
    # TODO: Change to two arrays, and build solution on termination
    # TODO: Reconsider the above, since the update_result! functino is nice
    res = ReparametrizationSolution(q, r, projector, maxiter)
    update_result!(res, l2_distance(q, r)^2, id)

    if verbosity == 1
        printheader()
        printfirst(l2_distance(q, r)^2)
    end

    ψ = id
    for i in 1:maxiter
        ∇E = l2_gradient(q, r)
        dγ = project(∇E, projector) #, min(projector.dim, i))

        # Choose Step Size
        εmax = max_step_length(dγ, alpha=0.999)
        ε = backtracking(q, r, dγ, εmax / 5, ρ=0.9, c=0.8, verbose=(verbosity >= 2), maxiter=300)

        dγ.W .*= ε
        γ(x) = x - dγ(x)

        update_result!(res, l2_distance(q, r)^2, γ)

        # Compute gradient norm and relative error to check for termination.
        gradnorm = l2_norm(dγ)
        relerror = relative_error(res.errors[i+1], res.errors[i])

        if verbosity >= 1
            printline(i, l2_distance(q, r)^2, εmax, ε, gradnorm, relerror)
        end

        if gradnorm < gtol || relerror < rtol
            break
        end

        # Update Parametrization
        r = Q_reparametrization(r, γ)
        ψ = ψ ∘ γ
    end

    # Print closing line to make results prettier
    if verbosity >= 1
        printfooter()
    end
    return res
end

""" Update result obejct with the current iterations error and
reparametrization"""
function update_result!(res::ReparametrizationSolution, error, reparam)
    push!(res.errors, error)
    push!(res.reparams, reparam)
end


struct ReparametrizationSolution{T1<:Function, T2<:Function, TP<:AbstractProjector}
    q::T1
    r::T2
    projector::TP
    errors::Vector{Float64}
    reparams::Vector{Function}
    maxiter::Int

    function ReparametrizationSolution(q::T1, r::T2, projector::TP, maxiter::Int) where {T1<:Function, T2<:Function, TP<:AbstractProjector}
        errors = Vector{Float64}()
        reparams = Vector{Function}()
        new{T1, T2, TP}(q, r, projector, errors, reparams, maxiter)
    end
end



function l2_gradient(q, r)
    # Get the derivatives of the curves q, r, and compute the gradient of the cost function
    qdt(t) = ForwardDiff.derivative(q, t)
    rdt(t) = ForwardDiff.derivative(r, t)
    function (t)
        return r(t)⋅qdt(t) - q(t)⋅rdt(t)  # /diff_norm
    end
end

function accumulate_composition(funcvec)
    n = length(funcvec)
    out = Vector{Function}(undef, n)
    out[1] = funcvec[1]
    for i in 2:n
        out[i] = out[i-1] ∘ funcvec[i]
    end
    return out
end


function relative_error(f1, f0)
    return abs((f1 - f0) / f0)
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
