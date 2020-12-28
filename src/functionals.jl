function pointwise_distance(f, g)
    function (x)
        return vecnorm(f(x) - g(x))
    end
end


function pointwise_product(f, g)
    function (x)
        f(x) ⋅ g(x)
    end
end


function pointwise_square(f)
    function (x)
        return f(x)^2
    end
end

# function l2_norm(f; I::AbstractIntegrator=DefaultIntegrator)
#     integrand = pointwise_square(f)
#     return sqrt(integrate(integrand, I))
# end


# function l2_distance(f, g; I::AbstractIntegrator=DefaultIntegrator)
#     integrand = pointwise_square(pointwise_distance(f, g))
#     return sqrt(integrate(integrand, I))
# end


# function l2_inner_product(f, g; I::AbstractIntegrator=DefaultIntegrator)
#     integrand = pointwise_product(f, g)
#     return integrate(integrand, I)
# end

function l2_norm(f; I::AbstractIntegrator=DefaultIntegrator)
    return I(f, L2Norm())
end

function l2_distance(f, g; I::AbstractIntegrator=DefaultIntegrator)
    return I(f, g, L2Distance())
end

function l2_inner_product(f, g; I::AbstractIntegrator=DefaultIntegrator)
    return I(f, g, L2InnerProduct())
end


""" Compute the Palais Metric to be used later."""
function palais_inner_product(f, g; I::AbstractIntegrator=DefaultIntegrator)
    return error("Deprecated.")
    fdx(x) = derivative(f, x)
    gdx(x) = derivative(g, x)
    integrand = pointwise_product(fdx, gdx)
    return f(0) * g(0) + integrate(integrand, I)
end