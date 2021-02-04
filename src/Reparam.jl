module Reparam

# Alternative Integrators
include("integrators.jl")
export DefaultIntegrator, integrate, GaussLegendre

include("integrators2D.jl")
export GaussLegendre2D

# And the bases for subspaces to project onto
include("basis.jl")
export JacobiBasis, FourierSineBasis, GaussianRadialBasis
export PalaisBasis

include("basis2D.jl")
export FourierVectorBasis

# Function interpolation methods
include("interpolation.jl")
export RBFInterpolator, interpolate

# Include function projection methods 
include("projection.jl")
export FourierProjector, JacobiProjector, PalaisProjector

# Create Diffeomorphism Struct
include("diffeomorphism.jl")
export Diffeomorphism, Reparametrization

# Curve transformations
include("transform.jl")
export Q_transform, Q_reparametrization, Qmap, Qmap2D

# Linesearch and backtracking algorithm
include("linesearch.jl")
export backtracking, BacktrackConfig

# Reparametrization Algorithm Interface
include("reparametrization.jl")
export reparametrize, accumulate_composition, l2_gradient

# Plotting interfaces for curves. 
include("plotters.jl")
export plot_curve, plot_curve!

# Plotting for surfaces
#include("visual_surface.jl")
end #module"
