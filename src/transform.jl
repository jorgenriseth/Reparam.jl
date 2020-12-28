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


struct FiniteDifference; h::Float64; end

function Q_transform(c, FD::FiniteDifference)
    function (x)
        return sqrt(norm((c(x+FD.h) - c(x-FD.h))/(2*FD.h))) * c(x)
    end
end


function Q_reparametrization(q, γ, FD::FiniteDifference)
    function (x)
        return √norm((γ(x+FD.h) - γ(x-FD.h))/(2* FD.h)) * q(γ(x))
    end
end