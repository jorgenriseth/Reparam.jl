using Plots


# Define a simple plotting function for plotting curves in the plane.
function plot_curve(c; npoints=101, kwargs...)
    X = range(0, 1, length=npoints)
    cx = [c(xi)[1] for xi in X]
    cy = [c(xi)[2] for xi in X]
    plot(cx, cy; kwargs...)
end

# Define a simple plotting function for plotting curves in the plane.
function plot_curve!(c; npoints=101, kwargs...)
    X = range(0, 1, length=npoints)
    cx = [c(xi)[1] for xi in X]
    cy = [c(xi)[2] for xi in X]
    plot!(cx, cy; kwargs...)
end

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


"""
ALL FUNCTIONS BELOW ARE PROBABLY TEMPORARY
"""
function plot_reparametrization(c1, c2, γ; npoints=51, kwargs...)
    q = Q_transform(c1)
    r = Q_transform(c2)
    
    p1 = plot_curve(c1, label="\$c_1\$", shape=:circle)
    plot_curve!(c2 ∘ γ, label="\$c_2\$", shape=:circle)
    
    p2 = plot_curve(q, label="\$c_1\$")
    plot_curve!(Q_reparametrization(r, γ), label="\$c_2\$")
    plot(p1, p2)
end

function plot_comparison(r, label; X=0:0.05:1, kwargs...)
    plot_curve(q, X, color="black", label="q", shape=:circle; kwargs...)
    plot_curve!(r, X, label=label, shape=:circle)
end