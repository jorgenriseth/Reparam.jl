module Reparam

# Write your package code here.
include("interpolation/lagrange.jl")
export CreateLagrangeInterpolator, CreatePiecewiseLagrangeInterpolator

include("interpolation/util.jl")
export compute_error_2_norm, compute_error_max_norm, square_dist_sum
export create_nodes_chebyshev, create_nodes_equidistant

include("interpolation/gradient_descent.jl")
export GradientDescent

include("interpolation/rbf.jl")
export CreateRbfInterpolator

end #module
