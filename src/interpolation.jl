
""" Constructor for gaussian radial basis function with given center and shape parameter """
function gaussian(center, param)
    function (x)
        return exp(-(param * (x - center))^2)
    end
end


""" Constructor for radial bump function with given center and shape parameter """
function bump(center, param)
    function (x)
        if abs(x - center) < 1. / param
            return exp(-1.0 / (1.0-(param*(x - center))^2)) # * ℯ   # (scaled or not)
        end
        return 0.
    end
end

""" Function for building a set of interpolation basis functions of type rbf (gaussian or bump).
Functions are centered at the elements in nodes, and all share a single commoon parameter."""
function basis_constructor(rbf, nodes, param)
    return [rbf(xi, param) for xi in nodes]
end


""" Create the interpolation matrix of the function values of each basis function evaluated at 
the center nodes of the other basis functions""" 
function interpolation_matrix(nodes, basis)
    return [bi(xi) for xi in nodes, bi in basis]
end


""" Create the interpolation load vector, which should be the function values on internal nodes, and zero
on the boundaries."""
function load_vector(f, nodes)
    b = zeros(length(nodes))
    b[2:end-1] = [f(xi) for xi in nodes[2:end-1]]
    return b
end


""" Function Builder. Takes a vector of basis functions, and a set of weights,
and return a function taking the linear combinations of this at x."""
function build_function(weights, basis)
    @assert length(weights) == length(basis) "weights and basis needs to be same length"
    N = length(weights)
    function (x)
        out = 0.0
        for i in 1:N
            out += weights[i] * basis[i](x)
        end
        return out
    end
end



""" Interpolate the function f, using N basis elements of type rbf"""
function interpolate(f, N; rbf=gaussian, param=nothing)
    # Use standard parameter ε = 1 / h if none is specified
    @assert rbf in [gaussian, bump]
    @assert N > 1 "Should have a minimum of 2 nodes."

    if param === nothing
        if rbf === gaussian
            param = (N-1)
        else rbf === bump
            param = (N-1) * 2 * sqrt(log(2 / (ℯ-1)) / (1 + log(2/ (ℯ-1))))
        end
    end
    
    # Create equidistant nodes over [0, 1]
    nodes = range(0, 1, length=N)

    nodes = create_nodes_chebyshev(0, 1, N)

    # Create basis, and the linear system to find interpolation weights
    basis = basis_constructor(rbf, nodes, param)
    A = interpolation_matrix(nodes, basis)
    b = load_vector(f, nodes)

    # Solve the linear system to get the weigths
    weights = A \ b

    return build_function(weights, basis)
end


"""
Create a grid of n+1 Chebyshev points on the interval [a, b]. This should
give the intpolation polybnomial minimizing the max-norm over the interval.
"""
function create_nodes_chebyshev(a, b, n)
    # Get index-row (reversed for increasing nodes)
    j = n:-1:0

    # Get standardized cheby-nodes, then map to [a, b]
    nodes = cos.((j.+0.5) * pi / (n+1))
    return 0.5 * (b - a) .* nodes .+ 0.5 * (b + a)
end