using Reparam
using Test

f1(x) = 1 + x^2 - 3x^3
f2(x) = 1 / (1 + x^2)

X = range(-1, 1, length=3)

r1 = CreateRbfInterpolator(f1, X, 10)
r2 = CreateRbfInterpolator(f2, X, ones(length(X)))

@test all(r1.(X) ≈ f1.(X))
@test all(r2.(X) ≈ f2.(X))
