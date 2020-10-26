using SpecialFunctions: gamma
abstract type BasisFunction end

""" Callable struct evaluating a weighted sum of basis functions."""
struct BasisExpansion{TF <: Real, TB <: BasisFunction}
    W::Vector{TF}
    B::Vector{TB}
    dim::Int

    """ Constructor for BasisExpansion, taking a weight vector and a vector of
    basis functions."""
    function BasisExpansion(W::Vector{TF}, B::Vector{TB}) where {TF <: Real, TB <: BasisFunction}
        @assert length(W) == length(B) "Weights and Basis should be of same length"
        new{TF, TB}(W, B, length(W))
    end
end

""" Function call of BasisExpansion. Takes the weighted sum of the
basisfunctions stored in the object."""
function (BE::BasisExpansion)(x)
    out = 0.0
    for i in 1:BE.dim
        out += BE.W[i] * BE.B[i](x)
    end
    return out
end

""" Gaussian Radial Basis function

    r(x; ε, c) = exp(-ε*(x-c)^2 )

"""
struct GaussianRadialBasisFunction{T<:Real} <: BasisFunction
    center::T
    param::T
end

# Alternate constructor with promotion
GaussianRadialBasisFunction(c, p) = GaussianRadialBasisFunction(promote(c, p)...)

""" Function evaluation of a gaussian radial basis function """
function (g::GaussianRadialBasisFunction)(x)
    return exp(-(g.param * (x - g.center))^2)
end


struct FourierSineBasisFunction <: BasisFunction
    n::Int
end

function (f::FourierSineBasisFunction)(x)
    return √2 * sin(π * f.n * x)
end

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

""" Function Evaluation of PalaisBasisFunction """
function (p::PalaisBasisFunction)(x)
    if p.label == "sin"
        return sin(2π * p.n * x) / (√2 * π * p.n)
    end
    return (cos(2π * p.n * x) - 1) / (√2 * π * p.n)
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

struct JacobiBasisFunction <: BasisFunction
    C::Vector{Float64}
    n::Int

    function JacobiBasisFunction(n)
        # Compute Coefficients in the jacobi polynomial
        Cn = C(n)
        Bn = [Float64(bi / sqrt(Cn)) for bi in B(n)]
        return new(Bn, n)
    end
end

function (J::JacobiBasisFunction)(x)
    out = 0.0
    for m in 0:J.n
        out += J.C[m+1] * (x-1)^m
    end
    return out * x * (1-x)
end

# Various function constructing different function bases.
JacobiBasis(N::Int) = [JacobiBasisFunction(n) for n in 0:N-1]
PalaisBasis(N::Int) = [PalaisBasisFunction(n) for n in 1:N]
FourierSineBasis(N::Int) = [FourierSineBasisFunction(n) for n in 1:N]

function GaussianRadialBasis(N::Int; param=nothing)
    if param === nothing
        param = convert(Float64, N-1)
    end
    X = range(0, 1, length=N)
    return [GaussianRadialBasisFunction(xi, param) for xi in X]
end
