using LinearAlgebra
using Printf

"""Struct to hold constants for use in the BisecLinesearch"""
struct LineSearchConfig
    maxiter::Int
    c1::Float64
    c2::Float64
end

"""Set the defaultconfig to be used by the BisectLineSearchAlgorithm"""
const LSBaseConfig = LineSearchConfig(100, 0.1, 0.9)

"""
Bisection linesearch to find a step length "a" satisfying the weak wolfe
conditions, for use direction search optimization algorithms such as gradient
descent.
"""
function BisectLineSearch(f, df, p, x0;
    TOL=1e-6, verbose=false, config=LSBaseConfig)
    # Rename constants for simplicity
    c1, c2, maxiter = config.c1, config.c2, config.maxiter

    # Init steplength and bounds
    amin = 0.
    amax = Inf
    a = 1.

    # Initial step
    x1 = x0 + a * p
    f0 = f(x0)
    df0p = df(x0) ⋅ p

    iter = 1
    if verbose
        println("="^20, "[LineSearch] START", "="^20)
        @printf "[LineSearch] Iter: %3d | a = %12.5e | f(x) = %12.f\n" iter a f(x0)
    end

    while iter < maxiter# && a * norm(p) > TOL
        iter += 1

        # No Sufficient decrease
        if f(x1) > f(x0) + c1 * a * df0p
            amax = a
            a = 0.5 * (amin + amax)

        # No Curvature condition
        elseif df(x1) ⋅ p < c2 * df0p
            amin = a
            if amax == Inf
                a = 2a
            else
                a = (amin + amax) / 2
            end

        # Weak Wolfe Satisfied
        else
            break
        end

        # Prepare for new iteration
        x1 = x0 + a * p
        if verbose
            @printf "[LineSearch] Iter: %3d | a = %12.5e | f(x) = %12.f\n" iter a f(x1)
        end
    end
    if iter == maxiter
        println("[BisectLineSearch] Max Iter Reached.")
    elseif a < TOL
        println("[BisectLineSearch] Step size below tolerance.")
    end

    if verbose
        println("="^20, "[LineSearch] STOP", "="^20)
    end

    return a
end

"""
Gradient descent algorithm for optimizing a function f, with derivative/gradient
df, using a bisection linesearch to determine step length in the direction of
the negative gradient.

Terminates after maxiter, or if any step smaller than TOL is taken (i.e.
norm(x1-x0) < TOL.)
"""
function GradientDescent(f, df, x0; gtol=1e-10, xtol=1e-10, verbosity=0,
    maxiter=500, lsconfig=LSBaseConfig)
    # Initialize values
    p = - df(x0)
    a = Inf
    x1 = Array{Float64}(undef, length(x0))

    # Store convergence
    history = Array{Float64}(undef, maxiter)
    history[1] = f(x0)

    # Log if verbose
    if verbosity >= 1
        @printf "%s" "" # Flush stdout (due to bug in Juno)
        @printf "[GD] Iter = %4d | f(x)=%16.10f | |df| = %15.10f | a = %12.5e\n" 0 f(x0) norm(p) a
    end

    iter = 1
    while norm(p) > gtol && a > xtol && iter < maxiter
        # Normalize search direction
        p = p / norm(p)

        # Perform linesearcvh in direction of p
        v = (verbosity >= 2) # Check verbosity level
        a = BisectLineSearch(f, df, p, x0, TOL=xtol, verbose=(verbosity >= 2),
                config=lsconfig)

        # Log if verbose
        if verbosity >= 1
            @printf "[GD] Iter = %4d | f(x)=%16.10f | |df| = %15.10f | a = %12.5e\n" iter f(x0) norm(df(x0)) a
        end

        # Update steps
        x1 = x0 + a * p
        x0 = x1
        history[iter+1] = f(x1)

        # Prepare for next iteration
        iter += 1
        p = - df(x0)
    end

    if iter == maxiter
        println("[GD] Max iter reached. Did not converge.")
    end
    return x1, history[1:iter]
end
