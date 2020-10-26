using QuadGK
using ForwardDiff: derivative

""" Construct function taking pointwise difference between f and g"""
function pointwise_distance(f, g)
    function (x)
        return norm(f(x) - g(x))
    end
end


""" Construct function taking piecewise product of f and g """
function pointwise_product(f, g)
    function (x)
        f(x) ⋅ g(x)
    end
end

""" Construct function taking piecewise square of  f"""
function pointwise_square(f)
    function (x)
        return norm(f(x))^2 
    end
end


""" Create a general function for performing integration of f (over 0, 1)"""
function integrate(f, args...; method=quadgk_adaptive, kwargs...)
    return method(f, args...; kwargs...)
end


""" General purpose adaptive quadrature using the QuadGK package """
function quadgk_adaptive(f)
    return quadgk(f, 0, 1)[1]
end


""" Compute L2-norm of function."""
function l2_norm(f; integrator=quadgk_adaptive)
    integrand = pointwise_square(f)
    return sqrt(integrate(integrand, method=integrator))
end


""" Compute L2-distance between two funciton f and g"""
function l2_distance(f, g; integrator=quadgk_adaptive)
    integrand = pointwise_square(pointwise_distance(f, g))
    return sqrt(integrate(integrand, method=integrator))
end


""" Compute the L2 inner product between two function f and g""" 
function l2_inner_product(f, g; integrator=quadgk_adaptive)
    integrand = pointwise_product(f, g)
    return integrate(integrand, method=integrator)
end


""" Compute the Palais Metric to be used later."""
function palais_inner_product(f, g; integrator=quadgk_adaptive)
    fdx(x) = derivative(f, x)
    gdx(x) = derivative(g, x)
    integrand = pointwise_product(fdx, gdx)
    return f(0) * g(0) + integrate(integrand, method=integrator)
end