using Reparam
using Test


# Test Gradient Descent Optimizer
f(x) = 2 * (x[1] - 1)^2 + x[2]^2
df(x) = [4 * (x[1] - 1), 2 * x[2]]
x0 = randn(2)

x, h = GradientDescent(f, df, x0, verbosity=0, gtol=1e-12, xtol=1e-10)

@test f(x) < 1e-10