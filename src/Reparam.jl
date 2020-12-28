module Reparam

# Alternative Integrators
include("integrators.jl")
export DefaultIntegrator, integrate

# Curve transformations
include("transform.jl")
export Q_transform, Q_reparametrization


# Integration and inner product fo functions
include("functionals.jl")
export l2_norm, l2_distance, l2_inner_product

# And the bases for subspaces to project onto
include("basis.jl")
export JacobiBasis, FourierSineBasis, GaussianRadialBasis

# Function interpolation methods
include("interpolation.jl")
export RBFInterpolator, interpolate

# Include function projection methods 
include("projection.jl")
export FourierProjector, JacobiProjector

# Linesearch and backtracking algorithm
include("linesearch.jl")
export backtracking, BacktrackConfig

# Reparametrization Algorithm Interface
include("reparametrization.jl")
export reparametrize, accumulate_composition, l2_gradient

# Plotting interfaces for curves. 
include("plotters.jl")
export plot_curve, plot_curve!


module Surfaces
include("integrators.jl")
export GaussLegendre
end

end #module"