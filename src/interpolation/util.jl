"""
Create a grid of n+1 equidistant points on the interval [a, b],
including endpoints.
"""
function create_nodes_equidistant(a, b, n)
    return collect(range(a, b, length=n+1))
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


"""
Compute the sum of the squared distances of f and g on the points of X.
"""
function square_dist_sum(f, g, X)
    # Compute sum
    summ = 0
    for xi in X
        summ += (f(xi) - g(xi))^2
    end
    return summ
end

"""
Approximate the L2-distance between f and g, defined over the interval [a,b]
by a rectangle sum with N elements.
"""
function compute_error_2_norm(f, g, a, b, N)
    X = range(a, b, length=N+1)
    return sqrt((b - a) / N * square_dist_sum(f, g, X))
end


"""
Approximate the max-norm between f and g, by taking the maximum value over
a fine grid of N+1 points over [a, b].
"""
function compute_error_max_norm(f, g, a, b, N)
    X = range(a, b, length=N+1)
    return maximum(abs.(f.(X) - g.(X)))
end
