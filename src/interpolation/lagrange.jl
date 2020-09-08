using LinearAlgebra

"""
Creates the k-th lagrange basis function for a vector X of interpolation
nodes
"""
function LagrangeBasis(X, k)
    # Start with computing basis constant
    Ck = 1.0
    for xi in X[1:k-1]
        Ck *= X[k] - xi
    end
    for xi in X[k+1:end]
        Ck *= X[k] - xi
    end

    # Now construct the function
    function Lk(x)
        out = 1.0
        for xi in X[1:k-1]
            out *= x - xi
        end
        for xi in X[k+1:end]
            out *= x - xi
        end
        return out / Ck
    end
    return Lk
end

"""
Create the Lagrange polynomial interpolating the function f at
the points in X.
"""
function CreateLagrangeInterpolator(f, X)
    basis = [LagrangeBasis(X, i) for i = 1:length(X)]
    return x -> f.(X) ⋅ [b(x) for b in basis]
end


"""
Create a piecewise polynomial over the interval [a, b] by creating a
Lagrange polynomial of degree n with equidistant nodes over each of the
subintervals [v[k], v[k+1]] with a = V[1] < V[2] < ... V[K] = b.
"""
function CreatePiecewiseLagrangeInterpolator(f, V, n)
    K = length(V) - 1
    polynomials = Array{Function}(undef, K)
    for i in 1:K
        nodes = create_nodes_equidistant(V[i], V[i+1], n)
        polynomials[i] = CreateLagrangeInterpolator(f, nodes)
    end

    function P(x)
        if x <= V[1]
            i = 1
        elseif x >= V[end]
            i = K
        else
            i = argmax((x .- V) .<= 0.) - 1
        end
        return polynomials[i](x)
    end
    return P
end
