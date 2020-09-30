using LinearAlgebra: norm
using ForwardDiff: gradient

# Define the q-transform for curves (with derivative given)
function Q_transform(c, cdt)
    return t -> sqrt(norm(cdt(t))) * c(t)
end

# Define the q-transform for curves (without derivative given)
function Q_transform(c)
    cdt(x) = gradient(c, x)
    return t -> sqrt(norm(cdt(t))) * c(t)
end
