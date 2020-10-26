module Reparam


# Curve transformations
include("transform.jl")
export Q_transform, Q_reparametrization

# Alternative Integrators
include("integrators.jl")
export DefaultIntegrator

# Integration and inner product fo functions
include("functionals.jl")
export l2_norm, l2_distance

# Linesearch and backtracking algorithm
include("linesearch.jl")
export backtracking, BacktrackConfig

# And the bases for subspaces to project onto
include("basis.jl")
export JacobiBasis, PalaisBasis, FourierSineBasis, GaussianRadialBasis

# Include function projection methods 
include("projection.jl")
export FourierSineProjector, JacobiProjector, PalaisProjector, GaussianInterpolationProjector

# And some plotting interfaces for curves. 
include("plotters.jl")
export plot_curve, plot_curve!

# Reparametrization Algorithm Interface
include("reparametrization.jl")
export reparametrize, accumulate_composition, l2_gradient


end #module