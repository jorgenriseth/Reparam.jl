using ForwardDiff: derivative
using LinearAlgebra: norm


function Q_transform(c)
    cdt(t) = derivative(c, t)
    function (t)
        return sqrt(norm(cdt(t))) * c(t)
    end
end


function Q_reparametrization(q, γ)
    γdt(t) = derivative(γ, t)
    function (t)
        return √(norm(γdt(t))) * q(γ(t))
    end
end