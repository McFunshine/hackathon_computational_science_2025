# Extended bottom topography generator for 1D shallow water equations
using Plots, Parameters, Random

# --- Enhanced Parameter Setup with Multiple Bottom Options ---
function make_all_topographies()
    # Basic parameters
    g = 9.81          # gravitational acceleration
    N = 200           # grid points
    xmax = 5.0        # domain length
    x = range(0.0, xmax, length=N)  # spatial grid
    D = 10.0          # domain depth
    
    # Create a dictionary of different bottom topographies
    bottoms = Dict()
    
    # 1. Flat bottom
    bottoms["flat"] = fill(-D, N)
    
    # 2. Original wavy bottom
    bottoms["wavy"] = -D .+ 0.4 .* sin.(2π .* x ./ xmax .* (N-1)/N .* 5)
    
    # 3. Sudden drop (step function)
    bottoms["sudden_drop"] = begin
        zb = fill(-D, N)
        drop_start = div(N, 3)
        drop_end = div(2*N, 3)
        zb[drop_start:drop_end] .= -D - 3.0  # 3m deeper section
        zb
    end
    
    # 4. Triangle mountain
    bottoms["triangle"] = begin
        zb = fill(-D, N)
        peak_idx = div(N, 2)
        peak_height = -D + 6.0  # 6m high mountain
        for i in 1:N
            if abs(i - peak_idx) <= div(N, 6)  # Triangle base width
                height_factor = 1.0 - abs(i - peak_idx) / (div(N, 6))
                zb[i] = -D + peak_height * height_factor
            end
        end
        zb
    end
    
    # 5. Multiple bumps
    bottoms["bumps"] = begin
        zb = fill(-D, N)
        # Add 3 Gaussian bumps
        for center in [0.2, 0.5, 0.8]
            center_idx = round(Int, center * N)
            for i in 1:N
                distance = abs(i - center_idx) / N
                zb[i] += 2.0 * exp(-50 * distance^2)
            end
        end
        zb
    end
    
    # 6. Staircase
    bottoms["staircase"] = begin
        zb = fill(-D, N)
        step_size = div(N, 5)
        for step in 1:4
            start_idx = step * step_size
            end_idx = min((step + 1) * step_size, N)
            zb[start_idx:end_idx] .= -D + step * 1.5  # Each step is 1.5m higher
        end
        zb
    end
    
    # 7. Sinusoidal canyon
    bottoms["canyon"] = begin
        zb = -D .+ 3.0 .* sin.(4π .* x ./ xmax) .* exp.(-((x .- xmax/2) ./ (xmax/4)).^2)
    end
    
    # 8. Random rocky bottom
    bottoms["rocky"] = begin
        zb = fill(-D, N)
        # Add random perturbations
        Random.seed!(42)  # For reproducibility
        for i in 1:N
            zb[i] += 0.8 * (rand() - 0.5) * sin(20π * x[i] / xmax)
        end
        zb
    end
    
    # 9. Cliff with plateau
    bottoms["cliff"] = begin
        zb = fill(-D, N)
        cliff_start = div(2*N, 5)
        cliff_end = div(3*N, 5)
        # Smooth transition to avoid numerical issues
        for i in cliff_start:cliff_end
            transition = (i - cliff_start) / (cliff_end - cliff_start)
            zb[i] = -D + 4.0 * transition  # 4m high plateau
        end
        zb[cliff_end+1:end] .= -D + 4.0
        zb
    end
    
    # 10. Volcanic island (bell curve)
    bottoms["volcano"] = begin
        zb = fill(-D, N)
        center = xmax / 2
        for i in 1:N
            distance = (x[i] - center) / (xmax * 0.3)
            zb[i] = -D + 8.0 * exp(-distance^2)  # 8m high volcanic peak
        end
        zb
    end
    
    return (; g, N, x, D, xmax, bottoms)
end

# --- Visualization Function ---
function visualize_all_bottoms(params)
    @unpack x, bottoms, D = params
    
    # Create a comprehensive plot
    n_plots = length(bottoms)
    n_cols = 3
    n_rows = ceil(Int, n_plots / n_cols)
    
    p = plot(layout=(n_rows, n_cols), size=(1200, 300 * n_rows))
    
    plot_idx = 1
    for (name, zb) in bottoms
        if plot_idx <= n_plots
            # Plot the bottom profile
            plot!(p, x, zb, 
                  subplot=plot_idx, 
                  linewidth=3, 
                  color=:saddlebrown,
                  fillto=minimum(zb)-1, 
                  fillcolor=:saddlebrown,
                  alpha=0.7,
                  title=uppercasefirst(name),
                  label="")
            
            # Add a water surface for reference (calm water)
            water_surface = fill(0.0, length(x))
            plot!(p, x, water_surface, 
                  subplot=plot_idx,
                  linewidth=2,
                  color=:blue,
                  label="",
                  linestyle=:dash)
            
            # Fill water area
            plot!(p, x, zb,
                  subplot=plot_idx,
                  fillto=water_surface,
                  fillcolor=:lightblue,
                  alpha=0.4,
                  linewidth=0,
                  label="")
            
            xlabel!(p, "Distance x [m]", subplot=plot_idx)
            ylabel!(p, "Elevation [m]", subplot=plot_idx)
            
            # Set consistent y-axis limits
            ylims!(p, minimum(zb) - 1, maximum(zb) + 2, subplot=plot_idx)
            
            plot_idx += 1
        end
    end
    
    # Remove empty subplots
    for i in plot_idx:n_rows*n_cols
        plot!(p, [], [], subplot=i, showaxis=false, grid=false)
    end
    
    plot!(p, plot_title="Bottom Topography Options for Shallow Water Simulation")
    
    # Save the plot
    savefig(p, "all_bottom_topographies.png")
    println("Saved all bottom topographies to: all_bottom_topographies.png")
    
    return p
end

# --- Individual Bottom Profile Visualizer ---
function visualize_single_bottom(bottom_name, params; add_disturbance=true)
    @unpack x, bottoms, N = params
    
    if !haskey(bottoms, bottom_name)
        println("Available bottoms: ", keys(bottoms))
        return nothing
    end
    
    zb = bottoms[bottom_name]
    
    p = plot(size=(1000, 600))
    
    # Plot bottom
    plot!(p, x, zb, 
          linewidth=4, 
          color=:saddlebrown,
          fillto=minimum(zb)-2, 
          fillcolor=:saddlebrown,
          alpha=0.8,
          label="Bottom Profile")
    
    # Add initial water disturbance if requested
    if add_disturbance
        # Same initial condition as in original code
        h_initial = 0.1 .* exp.(-100 .* ((x ./ maximum(x) .- 0.5) .* maximum(x)).^2)
        water_surface = h_initial .+ zb
        
        plot!(p, x, water_surface, 
              linewidth=3,
              color=:blue,
              label="Initial Water Surface")
        
        # Fill water area
        x_fill = vcat(x, reverse(x))
        y_fill = vcat(zb, reverse(water_surface))
        plot!(p, x_fill, y_fill, 
              seriestype=:shape,
              alpha=0.4,
              color=:lightblue,
              label="Water",
              linewidth=0)
    else
        # Just show calm water surface
        water_surface = fill(0.0, length(x))
        plot!(p, x, water_surface, 
              linewidth=2,
              color=:blue,
              linestyle=:dash,
              label="Calm Water Surface")
    end
    
    xlabel!(p, "Distance x [m]")
    ylabel!(p, "Elevation [m]")
    title!(p, "$(uppercasefirst(bottom_name)) Bottom Profile")
    
    # Save individual plot
    filename = "bottom_$(bottom_name).png"
    savefig(p, filename)
    println("Saved $(bottom_name) bottom profile to: $(filename)")
    
    return p
end

# --- Function to get bottom for simulation ---
function get_bottom_profile(bottom_name, params)
    @unpack bottoms = params
    
    if haskey(bottoms, bottom_name)
        return bottoms[bottom_name]
    else
        println("Bottom '$bottom_name' not found. Available options:")
        for name in keys(bottoms)
            println("  - $name")
        end
        return bottoms["flat"]  # Default fallback
    end
end

# --- Modified parameter function for original code ---
function make_parameters(bottom_type="flat")
    all_params = make_all_topographies()
    @unpack g, N, x, D, xmax, bottoms = all_params
    
    # Get the specified bottom
    zb = get_bottom_profile(bottom_type, all_params)
    
    tstart = 0.0
    tstop = 1.0
    
    return (; g, N, x, D, zb, tstart, tstop, xmax)
end

# # --- Demo Script ---
# function demo_all_bottoms()
#     println("🌊 Generating all bottom topographies...")
    
#     # Create all topographies
#     params = make_all_topographies()
    
#     # Visualize all bottoms in one plot
#     println("📊 Creating overview plot...")
#     overview_plot = visualize_all_bottoms(params)
    
#     # Create individual detailed plots for a few interesting ones
#     interesting_bottoms = ["triangle", "sudden_drop", "volcano", "canyon", "staircase"]
    
#     println("🎨 Creating detailed plots for interesting topographies...")
#     for bottom_name in interesting_bottoms
#         visualize_single_bottom(bottom_name, params, add_disturbance=true)
#     end
    
#     println("✨ All visualizations complete!")
#     println("\nAvailable bottom types for simulation:")
#     for name in keys(params.bottoms)
#         println("  - $name")
#     end
    
#     return params
# end

# --- Integration with original simulation code ---
function run_simulation_with_bottom(bottom_type="flat")
    """
    Run the shallow water simulation with a specific bottom type.
    Replace the make_parameters() call in your original code with this.
    """
    return make_parameters(bottom_type)
end

# # Run the demo
# demo_all_bottoms() 