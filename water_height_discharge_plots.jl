# Specialized plotting functions for water height and discharge visualization
using Plots, Parameters

"""
    plot_height_discharge_evolution(sol, params; save_plots=true)

Creates detailed plots of water height and discharge evolution over time.
Saves plots as PNG files if save_plots=true.
"""
function plot_height_discharge_evolution(sol, params; save_plots=true)
    @unpack N, x, zb, tstart, tstop, xmax = params
    
    # Time points for visualization
    t_plot = range(sol.t[1], sol.t[end], length=min(20, length(sol.t)))
    
    # Create figure with 3 subplots
    p1 = plot(layout=(3, 1), size=(1000, 1200))
    
    # Plot 1: Water height evolution over space and time
    colors = palette(:viridis, length(t_plot))
    for (i, t) in enumerate(t_plot)
        u_t = sol(t)
        h_t = u_t[1:N]
        plot!(p1, x, h_t, subplot=1, 
              label="t = $(round(t, digits=2))s", 
              color=colors[i],
              linewidth=2,
              alpha=0.8)
    end
    xlabel!(p1, "Distance x [m]", subplot=1)
    ylabel!(p1, "Water Height h [m]", subplot=1)
    title!(p1, "Water Height Evolution h(x,t)", subplot=1)
    
    # Plot 2: Discharge evolution over space and time
    for (i, t) in enumerate(t_plot)
        u_t = sol(t)
        q_t = u_t[N+1:2N]
        plot!(p1, x, q_t, subplot=2,
              label="t = $(round(t, digits=2))s",
              color=colors[i],
              linewidth=2,
              alpha=0.8)
    end
    xlabel!(p1, "Distance x [m]", subplot=2)
    ylabel!(p1, "Discharge q [m²/s]", subplot=2)
    title!(p1, "Discharge Evolution q(x,t)", subplot=2)
    
    # Plot 3: Phase space plot (h vs q) at specific locations
    locations = [div(N,4), div(N,2), 3*div(N,4)]  # Quarter, half, three-quarter points
    location_labels = ["x = $(round(x[loc], digits=2))m" for loc in locations]
    
    for (i, loc) in enumerate(locations)
        h_series = [sol(t)[loc] for t in sol.t]
        q_series = [sol(t)[N + loc] for t in sol.t]
        plot!(p1, h_series, q_series, subplot=3,
              label=location_labels[i],
              linewidth=2,
              marker=:circle,
              markersize=2)
    end
    xlabel!(p1, "Water Height h [m]", subplot=3)
    ylabel!(p1, "Discharge q [m²/s]", subplot=3)
    title!(p1, "Phase Space: q vs h at Different Locations", subplot=3)
    
    if save_plots
        savefig(p1, "height_discharge_evolution.png")
        println("Saved height and discharge evolution plot to: height_discharge_evolution.png")
    end
    
    return p1
end

"""
    plot_height_discharge_heatmaps(sol, params; save_plots=true)

Creates heatmap visualizations of water height and discharge over space and time.
"""
function plot_height_discharge_heatmaps(sol, params; save_plots=true)
    @unpack N, x, tstart, tstop = params
    
    # Create time-space grids for heatmaps
    n_time_points = min(100, length(sol.t))
    t_grid = range(sol.t[1], sol.t[end], length=n_time_points)
    
    # Extract height and discharge matrices
    H_matrix = zeros(n_time_points, N)
    Q_matrix = zeros(n_time_points, N)
    
    for (i, t) in enumerate(t_grid)
        u_t = sol(t)
        H_matrix[i, :] = u_t[1:N]
        Q_matrix[i, :] = u_t[N+1:2N]
    end
    
    # Create heatmap plots
    p2 = plot(layout=(2, 1), size=(1000, 800))
    
    # Height heatmap
    heatmap!(p2, x, t_grid, H_matrix, subplot=1,
             color=:blues,
             xlabel="Distance x [m]",
             ylabel="Time t [s]",
             title="Water Height h(x,t) Heatmap",
             colorbar_title="h [m]")
    
    # Discharge heatmap
    heatmap!(p2, x, t_grid, Q_matrix, subplot=2,
             color=:reds,
             xlabel="Distance x [m]",
             ylabel="Time t [s]",
             title="Discharge q(x,t) Heatmap",
             colorbar_title="q [m²/s]")
    
    if save_plots
        savefig(p2, "height_discharge_heatmaps.png")
        println("Saved height and discharge heatmaps to: height_discharge_heatmaps.png")
    end
    
    return p2
end

"""
    plot_height_discharge_detailed(sol, params; save_plots=true)

Creates a comprehensive set of height and discharge plots including:
- Evolution plots
- Heatmaps  
- Time series at key locations
- Statistical analysis
"""
function plot_height_discharge_detailed(sol, params; save_plots=true)
    println("Creating detailed height and discharge visualizations...")
    
    # Create evolution plots
    p1 = plot_height_discharge_evolution(sol, params; save_plots=false)
    
    # Create heatmaps
    p2 = plot_height_discharge_heatmaps(sol, params; save_plots=false)
    
    # Create time series analysis
    p3 = plot_time_series_analysis(sol, params; save_plots=false)
    
    if save_plots
        savefig(p1, "detailed_height_discharge_evolution.png")
        savefig(p2, "detailed_height_discharge_heatmaps.png") 
        savefig(p3, "detailed_time_series_analysis.png")
        
        println("Saved detailed height and discharge plots:")
        println("  - detailed_height_discharge_evolution.png")
        println("  - detailed_height_discharge_heatmaps.png")
        println("  - detailed_time_series_analysis.png")
    end
    
    return p1, p2, p3
end

"""
    plot_time_series_analysis(sol, params; save_plots=true)

Creates time series analysis plots at different spatial locations.
"""
function plot_time_series_analysis(sol, params; save_plots=true)
    @unpack N, x = params
    
    # Select key locations for analysis
    locations = [1, div(N,4), div(N,2), 3*div(N,4), N]
    location_names = ["Left boundary", "Quarter point", "Center", "Three-quarter", "Right boundary"]
    
    p3 = plot(layout=(2, 1), size=(1000, 600))
    
    # Height time series
    for (i, loc) in enumerate(locations)
        h_series = [sol(t)[loc] for t in sol.t]
        plot!(p3, sol.t, h_series, subplot=1,
              label="$(location_names[i]) (x=$(round(x[loc], digits=2))m)",
              linewidth=2)
    end
    xlabel!(p3, "Time t [s]", subplot=1)
    ylabel!(p3, "Water Height h [m]", subplot=1)
    title!(p3, "Water Height Time Series at Key Locations", subplot=1)
    
    # Discharge time series
    for (i, loc) in enumerate(locations)
        q_series = [sol(t)[N + loc] for t in sol.t]
        plot!(p3, sol.t, q_series, subplot=2,
              label="$(location_names[i]) (x=$(round(x[loc], digits=2))m)",
              linewidth=2)
    end
    xlabel!(p3, "Time t [s]", subplot=2)
    ylabel!(p3, "Discharge q [m²/s]", subplot=2)
    title!(p3, "Discharge Time Series at Key Locations", subplot=2)
    
    if save_plots
        savefig(p3, "time_series_analysis.png")
        println("Saved time series analysis to: time_series_analysis.png")
    end
    
    return p3
end

# Example usage functions
"""
    run_height_discharge_plots(solution, params)

Convenience function to run all height and discharge plotting functions.
"""
function run_height_discharge_plots(solution, params)
    println("Generating comprehensive height and discharge visualizations...")
    
    # Create individual plot types
    plot_height_discharge_evolution(solution, params)
    plot_height_discharge_heatmaps(solution, params)
    plot_time_series_analysis(solution, params)
    
    println("\nAll plots saved successfully!")
    println("Generated files:")
    println("  - height_discharge_evolution.png")
    println("  - height_discharge_heatmaps.png") 
    println("  - time_series_analysis.png")
end 
