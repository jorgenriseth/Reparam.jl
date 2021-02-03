using PyPlot

function plot_parametric_surface(func; n_points=101, axis_title="", axis=nothing, camera=(25, -135), colorfunc=nothing, rel_colorfunc=nothing, colormap="viridis", kwargs...)
    # Plot Data
    xs = ys = range(0, 1, length=n_points)

    F = [func([xi, yi]) for xi in xs, yi in xs]
    fx = [el[1] for el in F]
    fy = [el[2] for el in F]
    fz = [el[3] for el in F]

    if colorfunc !== nothing
        colmap = [colorfunc([xi, yi]) for xi in xs, yi in xs]
        # Check relative colofunc
        if rel_colorfunc !== nothing
            col_min, col_max = get_min_max_grid_value(rel_colorfunc)
            colmap = colmap / col_max  
        else
            col_min = minimum(colmap)
            col_max = maximum(colmap)
            colmap = colmap / col_max
        end
        
        kwargs = (facecolors = get_cmap(colormap)(colmap), kwargs...)
    end

    if axis === nothing
        fig = figure(figsize=(8, 8))
        plot_surface(fx, fy, fz, rcount=100, ccount=100; kwargs...)
        xlabel("x")
        ylabel("y")
        zlabel("z")
        title(axis_title)
        gca().view_init(camera...)
        return fig
    end

    axis.plot_surface(fx, fy, fz, rcount=200, ccount=200; kwargs...)
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_xlabel("x")
    axis.set_title(axis_title)
    axis.view_init(camera[1], camera[2])
    return axis
end

function plot_parametric_wireframe(func; n_points=21, axis_title="", axis=nothing, camera=(25, -135), kwargs...)
    # Plot Data
    xs = ys = range(0, 1, length=n_points)

    F = [func([xi, yi]) for xi in xs, yi in xs]
    fx = [el[1] for el in F]
    fy = [el[2] for el in F]
    fz = [el[3] for el in F]

    if axis === nothing
        fig = figure(figsize=(8, 8))
        plot_wireframe(fx, fy, fz; kwargs...)
        xlabel("x")
        ylabel("y")
        zlabel("z")
        title(axis_title)
        gca().view_init(camera...)
        return fig
    end

    axis.plot_wireframe(fx, fy, fz; kwargs...)
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_xlabel("x")
    axis.set_title(axis_title)
    axis.view_init(camera[1], camera[2])
    return axis
end

function plot_diffeomorphism(ψ; n_points=15, axis=nothing)
    xs = 0.:(1/n_points):1.
    
    if axis === nothing
        fig = figure()
        
        # Plot horizontal lines
        for xi in xs
            line_x, line_y = [xi for yi in xs], [yi for yi in xs]
            warped_line_x = [ψ([xi, yi])[1] for yi in xs]
            warped_line_y = [ψ([xi, yi])[2] for yi in xs]

            # plot(line_x, line_y, c="lightgrey", lw=0.7)
            plot(warped_line_x, warped_line_y, c="k")
        end
        
        # Plot Warped Lines Vertically
        for yi in xs
            line_x, line_y = [xi for xi in xs], [yi for xi in xs]
            warped_line_x = [ψ([xi, yi])[1] for xi in xs]
            warped_line_y = [ψ([xi, yi])[2] for xi in xs]

            # plot(line_x, line_y, c="lightgrey", lw=0.7)
            plot(warped_line_x, warped_line_y, c="k")
        end
        
        return fig
    end
    
    
        # Plot horizontal lines
    for xi in xs
        line_x, line_y = [xi for yi in xs], [yi for yi in xs]
        warped_line_x = [ψ([xi, yi])[1] for yi in xs]
        warped_line_y = [ψ([xi, yi])[2] for yi in xs]

        # axis.plot(line_x, line_y, c="lightgrey", lw=0.7)
        axis.plot(warped_line_x, warped_line_y, c="k")
    end

    # Plot Warped Lines Vertically
    for yi in xs
        line_x, line_y = [xi for xi in xs], [yi for xi in xs]
        warped_line_x = [ψ([xi, yi])[1] for xi in xs]
        warped_line_y = [ψ([xi, yi])[2] for xi in xs]

        # axis.plot(line_x, line_y, c="lightgrey", lw=0.7)
        axis.plot(warped_line_x, warped_line_y, c="k")
    end
    return axis
end

"""
Retrieve color scale
"""
function get_min_max_grid_value(func, n_points=101)
    xs = 0:(1/n_points):1
    A = [func([xi, yi]) for xi in xs, yi in xs]
    
    return minimum(A), maximum(A)
end


function add_point(point, axis; kwargs...)
    if length(point) == 2     
        axis.scatter([point[1]], [point[2]]; kwargs...)
    elseif length(point) == 3    
        axis.scatter([point[1]], [point[2]], [point[3]]; kwargs...)
    else
        println("Dim must be 2 or 3")
    end
end

function plot_contourf(func, axis, n_points=101; kwargs...)
    # Plot Data
    xs = ys = range(0, 1, length=n_points)

    F = [func([xi, yi]) for xi in xs, yi in xs]
    axis.contourf(xs, xs, F; kwargs...)
    return axis
end

