include("functionals.jl")

using SpecialFunctions: gamma

"""Computes the orthogonal projection coefficients for a function f in the direction
of each of the functions in the basis"""
function find_projection_weights(f, basis)
    return [l2_inner_product(f, bi) for bi in basis]
end


"""Orthogonal projection of f onto span(basis) with respect to the L2-metric""" 
function project(f, basis)
    weights = find_projection_weights(f, basis)
    return build_function(weights, basis)
end


"""Create a vector containing the fourier sine series with 
period up to N"""
function basis_fourier_sine(N)
    return [x -> sqrt(2) * sin(π*n*x) for n in 1:N]
end


""" Function to compute the normalizing constant of the n-th 
Jacobi-based polynomial"""
function C(n)
    return (2^5 / (2n+5))  * (gamma(n+3) * gamma(n+3)) / (32 * gamma(n+5) * factorial(big(n)))
end


""" Function to compute the coefficients of the n-th 
Jacobi-based polynomial"""
function B(n)
    return [gamma(n+3) * binomial(n, m) * gamma(n+m+5) / (factorial(big(n)) * gamma(n+5) * gamma(m+3)) for m in 0:n]
end


"""Construct the n-th jacobi-based polynomial. Note that 
the degree of the polynomial is 2 larger than n, n<2.
"""
function jacobi_polynomial(n)
    # Compute Coefficients in the jacobi polynomial
    Cn = C(n)
    Bn = [Float64(bi / sqrt(Cn)) for bi in B(n)]
    
    # Create function
    function (x)
        out = 0.0
        for m in 0:n
            out += Bn[m+1] * (x-1)^m
        end
        return out * x * (1-x)
    end
end


function basis_jacobi_polynomials(N)
    return [jacobi_polynomial(n) for n in 0:N-1]
end



""" Create Palais basis (currently not in use, but added for possible future used) """
function basis_palais(nmax)
    basis = []
    for n in 1:nmax
        if n % 2 == 1
            m = n ÷ 2 + 1
            push!(basis, x -> (cos(2π*m*x) - 1) / (√2π * m))
        else
            m = n ÷ 2
            push!(basis, x -> sin(2π*m*x) / (√2π * m))
        end
    end
    return basis
end