using SpecialFunctions: gamma
using Jacobi: jacobi
abstract type BasisFunction <: Function end

# Define a type to compute the weighted sum of basis functions in B, with weigths W.
struct BasisExpansion{T <: BasisFunction} <: Function
    W::Vector{Float64}
    B::Vector{T}
    dim::Int
end


# Constructor
function BasisExpansion(W,  B::Vector{T}) where {T <: BasisFunction}
    @assert length(W) == length(B) "Weights and Basis should be of same length"
    BasisExpansion(W, B, length(W))
end

# Callable BasisExpansion
function (BE::BasisExpansion)(x)
    out = zero(x)
    for i in 1:BE.dim
        out += BE.W[i] * BE.B[i](x)
    end
    return out
end


# Rescale the weights in a BasisExpansion
function rescale!(BE::BasisExpansion, a)
    BE.W .*= a
end


# Fourier Sine Basis function
struct FourierSineBasisFunction <: BasisFunction
    n::Int
end

# Make basis callable
function (f::FourierSineBasisFunction)(x)
    return √2 * sin(π * f.n * x)
end


"""
Jacobi-Polynomial Construtor Helper, for polynomial coefficient.
"""
function C(n)
    return (2^5 / (2n+5))  * (gamma(n+3) * gamma(n+3)) / (32 * gamma(n+5) * factorial(big(n)))
end


"""
"Jacobi-Polynomial-Type" Basis Functions. Weighted Jacobi Polynomial which is 
orhtonormal with respect to L2-metric, and vanishes at 0 and 1.
"""
struct JacobiBasisFunction <: BasisFunction
    n::Int
    Cn::Float64
end


# Constructor
JacobiBasisFunction(n) = JacobiBasisFunction(n, √C(n))

# Caller
function (J::JacobiBasisFunction)(x)
    return jacobi(2x-1, J.n, 2, 2) * x * (1-x) / J.Cn
end


"""
Trigonometric basis orthonormal with respect to a first-order Palais metric, 
with functions vanishing at 0 and 1.
"""
struct PalaisBasisFunction <: BasisFunction
    n::Int
    label::String

    function PalaisBasisFunction(m::Int, label::String)
        @assert label in ["cos", "sin"] "label must be of type 'sin' or 'cos'."
        new(m, label)
    end
end

# Alternative constructor with single argument, choosing type based on odd/even
function PalaisBasisFunction(n::Int)
    n % 2 == 0 ? PalaisBasisFunction(n ÷ 2, "sin") : PalaisBasisFunction(n ÷ 2 + 1, "cos")
end


""" 
Function Evaluation of PalaisBasisFunction
"""
function (p::PalaisBasisFunction)(x)
    if p.label == "sin"
        return sin(2π * p.n * x) / (√2 * π * p.n)
    end
    return (cos(2π * p.n * x) - 1) / (√2 * π * p.n)
end



"""
Create Vectors of length N of the different basisfunctions
"""
FourierSineBasis(N) = [FourierSineBasisFunction(n) for n in 1:N]
JacobiBasis(N) = [JacobiBasisFunction(n) for n in 0:N-1]
PalaisBasis(N) = [PalaisBasisFunction(n) for n in 1:N]

