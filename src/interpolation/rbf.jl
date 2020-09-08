"""
The exponential radial basis function
    ϕ(r) = exp(-εr²)
centered at center,
"""
function RbfExponential(center, ε)
    return x -> exp(-ε * abs(x - center)^2)
end


"""
Create radial basis function interpolating the function f in the points
given in X, where all basis function share the parameter ε.
"""
function CreateRbfInterpolator(f, X, param::Number)
    basis = [RbfExponential(xi, param) for xi in X]
    M = Symmetric([b(xj) for xj in X, b in basis])
    weights = M \ f.(X)
    return x -> weights ⋅ [b(x) for b in basis]
end


"""
Create radial basis function interpolating the function f in the points
given in X, where all basis function have their own parameter εi.
"""
function CreateRbfInterpolator(f, X, params::Array)
    @assert length(X) == length(params) "Params should either be single number, or an array of same length as X"

    basis = [RbfExponential(X[i], params[i]) for i in 1:length(X)]
    M = [b(xj) for xj in X, b in basis]
    weights = M \ f.(X)

    return x -> weights ⋅ [b(x) for b in basis]
end
