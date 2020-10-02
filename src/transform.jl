using LinearAlgebra: norm
using ForwardDiff: derivative

# Define the q-transform for curves (with derivative given)
function Q_transform(c, cdt)
    function q(t)
        return sqrt(norm(cdt(t))) * c(t)
    end
end

# Define the q-transform for curves (without derivative given)
function Q_transform(c)
    cdt(t) = derivative(c, t)
    function (t)
        return sqrt(norm(cdt(t))) * c(t)
    end
end
