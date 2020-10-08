using LinearAlgebra: ⋅

function l2_gradient(q, r)
    # Get the derivatives of the curves q, r, and compute the gradient of the cost function
    qdt(t) = ForwardDiff.derivative(q, t)
    rdt(t) = ForwardDiff.derivative(r, t)
    function (t)
        return r(t)⋅qdt(t) - q(t)⋅rdt(t)  # /diff_norm
    end
end
