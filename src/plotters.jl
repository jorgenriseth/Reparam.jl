using Plots

# Define a simple plotting function for plotting curves in the plane.
function plot_curve(c, X; kwargs...)
    cx = [c(xi)[1] for xi in X]
    cy = [c(xi)[2] for xi in X]
    plot(cx, cy; kwargs...)
end

# Define a simple plotting function for plotting curves in the plane.
function plot_curve!(c, X; kwargs...)
    cx = [c(xi)[1] for xi in X]
    cy = [c(xi)[2] for xi in X]
    plot!(cx, cy; kwargs...)
end