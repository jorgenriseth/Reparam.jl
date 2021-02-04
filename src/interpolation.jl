struct GaussianRadialBasisFunction <: BasisFunction
    center::Float64
    param::Float64
end


# Callable
function (g::GaussianRadialBasisFunction)(x)
    return exp(-(g.param * (x - g.center))^2)
end


struct Interpolator{T<:BasisFunction}
    basis::Vector{T}  # Interpolation basis functions.
    M::Matrix{Float64}  # Mass matrix Mij = <bi, bj> .
    b::Vector{Float64}  # Pre-allocated interpolation vector.
    w::Vector{Float64} # Pre-allocated solution vector.
    nodes::Vector{Float64}  # Interpolation nodes.
    dim::Int  # Number of basis functions.
end

# Constructor
function Interpolator(basis::Vector{T}) where {T<:BasisFunction}
    n = length(basis)
    M = [bi(bj.center) for bj in basis, bi in basis]
    nodes = [bi.center for bi in basis]
    return Interpolator(basis, M, zeros(n), zeros(n), nodes, n)
end


function GaussianRadialBasis(N)
    param = (N-1)
    return [GaussianRadialBasisFunction(xi, param) for xi in range(0, 1, length=N)]
end


function interpolate(f, IP::Interpolator)
    IP.b .= f.(IP.nodes)
    IP.w .= IP.M \ IP.b
    return BasisExpansion(IP.w, IP.basis)
end


RBFInterpolator(N) = Interpolator(GaussianRadialBasis(N))



""" 
Alternative Reparametrization with intermediate interpolation step. 
Not thoroughly tested.
"""
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