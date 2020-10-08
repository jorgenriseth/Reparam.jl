using LinearAlgebra: norm
using ForwardDiff: derivative

# Define the q-transform for curves (with derivative given)
function Q_transform(c, cdt)
    function (t)
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

function Q_reparametrization(q, γ)
    γdt(t) = ForwardDiff.derivative(γ, t)
    function (t)
        return √(norm(γdt(t))) * q(γ(t))
    end
end