include("functionals.jl")
""" Algorithm 3.1. Nocedal & Wright Numerical Optimization: Backtracking Line Search
Simple approach for choosing the step size in a line serach optimization algorithm. The algorithm 
either takes a selected step lengtho f alphamax, or a short enough step to satisfy sufficient decrease
condition with constant c."""
function backtracking(q, r, v, εmax; ρ=0.9, c=0.1, maxiter=100)
    # Initialize step length
    ε = εmax

    # Create function stepping along v
    γ(x) = x - ε * v(x)

    # Get initial values for sufficient decrease condition.
    E0 = l2_distance(q, r)^2
    v0 = l2_norm(v)^2
    
    l = make_weak_wolfe_line(E0, v0, c)
    
    # Initialize iteration and 
    iter = 0
    while l2_distance(q, r ∘ γ)^2 > l(ε) && iter < maxiter
        iter += 1
        ε *= ρ
        # @printf "Iter %4d: %10.5f vs. %10.5f\n" iter l2_distance(q, r ∘ γ)^2 l(ε)
    end

    # Warn user if backtracking didn't work
    if iter == maxiter
        println("[backtracking] Warning: Couldn't find sufficient decrease stepsize in $iter steps.")
    end

    return ε
end

""" Create the lower bound for cost functional for step in the directino of v to be accepted. """
function make_weak_wolfe_line(E0, v_norm, c)
    a = c * v_norm
    function (ε)
        return E0 - a * ε
    end
end


""" Find the maximum allowed step size away from the identity element for the function to
be monotonously increasing over [0, 1]"""
function max_step_length(v; Nfine=201, alpha=0.95)
    vdt(x) = ForwardDiff.derivative(v, x)
    vi = vdt.(range(0, 1, length=Nfine))
    return 0.95 / maximum(vi)
end