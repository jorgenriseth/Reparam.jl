using Reparam
using Test

f1(x) = 1 + x^2 - 3x^3
f2(x) = 1 / (1 + x^2)

X = range(-1, 1, length=4)

p1 = CreateLagrangeInterpolator(f1, X)
p2 = CreateLagrangeInterpolator(f2, X)


# Should interpolate f in the nodes
@test all(p1.(X) ≈ f1.(X))
@test all(p2.(X) ≈ f2.(X))

# Since the functions are polynomialsof sufficiently low degree,
# they should match everywhere
xs =  -1:0.1:1
@test all(p1.(xs) ≈ f1.(xs))
