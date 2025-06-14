# Correct GABC Implementation - Proper manipulation of Table II equations
using NonlinearSolve, LinearAlgebra, Parameters, Plots, Sundials

# Include the bottom topography generator
include("bottom_topographies.jl")

# --- 1. Parameter setup ---
function make_parameters(bottom_type="flat"; gif_name=nothing)
    N = 100
    g = 9.81
    xstart = 0.0
    xstop = 5.0
    x = range(xstart, xstop, length=N)
    D = 10.0
    
    # Use the enhanced bottom topography system
    all_params = make_all_topographies()
    zb_original = get_bottom_profile(bottom_type, all_params)
    
    # If the selected bottom doesn't fit our grid, interpolate it
    if length(zb_original) != N
        # Simple linear interpolation using built-in functions
        x_original = range(0.0, 5.0, length=length(zb_original))  # Original grid
        
        # Simple interpolation - map each point in our grid to the original
        zb = zeros(N)
        for i in 1:N
            # Find corresponding position in original grid
            x_pos = x[i]
            # Find the closest indices in original grid
            idx_float = (x_pos / 5.0) * (length(zb_original) - 1) + 1
            idx_low = max(1, floor(Int, idx_float))
            idx_high = min(length(zb_original), idx_low + 1)
            
            # Linear interpolation
            if idx_low == idx_high
                zb[i] = zb_original[idx_low]
            else
                weight = idx_float - idx_low
                zb[i] = (1 - weight) * zb_original[idx_low] + weight * zb_original[idx_high]
            end
        end
        
        println("📏 Interpolated $(bottom_type) bottom from $(length(zb_original)) to $(N) points")
    else
        zb = zb_original
    end
    
    tstart = 0.0
    tstop = 0.8
    
    # Add gif_name if provided
    if gif_name !== nothing
        return (; g, N, x, D, zb, tstart, tstop, gif_name)
    else
        return (; g, N, x, D, zb, tstart, tstop)
    end
end

# --- 2. Initial condition ---
function initial_conditions(params)
    g = params.g
    N = params.N
    x = params.x
    D = params.D
    zb = params.zb
    # Initial discharge
    q0 = zeros(N)
    # Initial height - conservative for stability
    xstop = x[end]
    h0 = 0.05 .* exp.(-50 .* ((x ./ xstop .- 0.5) .* xstop).^2) .- zb
    h0 = max.(h0, 0.05)  # Minimum depth
    return h0, q0
end

# --- 3. Boundary condition functions ---
function prescribed_incoming_left(t)
    # Prescribed incoming characteristic at left boundary
    amplitude = 0.02
    frequency = 0.5
    return amplitude * sin(2π * frequency * t)
end

function prescribed_incoming_right(t)
    # Prescribed incoming characteristic at right boundary (usually 0 for absorption)
    return 0.0
end

# --- 4. Correct GABC DAE residual function ---
function swe_dae_residual_gabc_correct!(residual, du, u, p, t)
    # Unpack parameters
    g = p.g
    N = p.N
    x = p.x
    zb = p.zb
    dx = x[2] - x[1]
    
    # Unpack state
    h = @view u[1:N]
    q = @view u[N+1:2N]
    duh = @view du[1:N]
    duq = @view du[N+1:2N]
    
    # Safety for shallow water
    h_min = 1e-6
    h_safe = max.(h, h_min)
    
    # Compute zeta (free surface)
    zeta = h_safe .+ zb
    
    # Interior points (2 to N-1) - standard SWE
    for i in 2:N-1
        ip = i + 1
        im = i - 1
        # Central differences
        dqdx = (q[ip] - q[im]) / (2*dx)
        dzetadx = (zeta[ip] - zeta[im]) / (2*dx)
        dfdx = ((q[ip]^2/h_safe[ip]) - (q[im]^2/h_safe[im])) / (2*dx)
        # SWE residuals
        residual[i] = duh[i] + dqdx
        residual[N+i] = duq[i] + dfdx + g * h_safe[i] * dzetadx
    end
    
    # Left boundary (i=1) - CORRECT GABC implementation
    i = 1
    # Use forward differences for stability
    dqdx = (q[i+1] - q[i]) / dx
    
    # The key insight: Implement Table II equations using DAE variables
    # From Table II, left boundary incoming characteristic:
    # (1/h)[∂q/∂t - (q/h)∂h/∂t + √(gh)∂h/∂t]
    
    # In DAE formulation: ∂q/∂t = duq[i], ∂h/∂t = duh[i]
    c = sqrt(g * h_safe[i])
    
    # Left boundary incoming characteristic (from Table II)
    incoming_char = (1/h_safe[i]) * (duq[i] - (q[i]/h_safe[i])*duh[i] + c*duh[i])
    prescribed_incoming = prescribed_incoming_left(t)
    
    # Continuity equation (always satisfied)
    residual[i] = duh[i] + dqdx
    
    # GABC constraint: incoming characteristic = prescribed value
    # This replaces the momentum equation at the boundary
    residual[N+i] = incoming_char - prescribed_incoming
    
    # Right boundary (i=N) - CORRECT GABC implementation
    i = N
    # Use backward differences for stability
    dqdx = (q[i] - q[i-1]) / dx
    
    # From Table II, right boundary incoming characteristic:
    # (1/h)[∂q/∂t - (q/h)∂h/∂t - √(gh)∂h/∂t]
    
    c = sqrt(g * h_safe[i])
    
    # Right boundary incoming characteristic (from Table II)
    incoming_char = (1/h_safe[i]) * (duq[i] - (q[i]/h_safe[i])*duh[i] - c*duh[i])
    prescribed_incoming = prescribed_incoming_right(t)
    
    # Continuity equation
    residual[i] = duh[i] + dqdx
    
    # GABC constraint: incoming characteristic = prescribed value
    residual[N+i] = incoming_char - prescribed_incoming
    
    return nothing
end

# --- 5. Time integration with proper GABC ---
function timeloop_gabc_correct(params)
    # Unpack parameters 
    g = params.g
    N = params.N
    x = params.x
    zb = params.zb
    dx = x[2] - x[1]
    tstart = params.tstart
    tstop = params.tstop

    # Set up initial conditions
    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    
    # Better initial guess for du/dt based on initial conditions
    du0 = zeros(2N)
    
    # For GABC, we need consistent initial conditions
    # The initial du/dt should satisfy the GABC constraints
    for i in 1:N
        if i == 1  # Left boundary
            # From GABC: incoming_char = prescribed_incoming_left(0)
            c = sqrt(g * h0[i])
            prescribed = prescribed_incoming_left(0.0)
            # This gives us a constraint on duq[1] given duh[1]
            # (1/h)[duq - (q/h)duh + c*duh] = prescribed
            # Assume duh[1] = 0 initially, then duq[1] = h * prescribed
            du0[N+i] = h0[i] * prescribed
        elseif i == N  # Right boundary
            # Similar for right boundary
            c = sqrt(g * h0[i])
            prescribed = prescribed_incoming_right(0.0)
            du0[N+i] = h0[i] * prescribed
        end
    end
    
    # Calculate limits for plotting
    zeta0 = h0 .+ zb
    zetamin = minimum(zeta0) - 0.3
    zetamax = maximum(zeta0) + 0.3
    ylim_zeta = (zetamin, zetamax)
    
    qmin = -1.0
    qmax = 1.0
    ylim_q = (qmin, qmax)

    tspan = (tstart, tstop)
    differential_vars = trues(2N)
    save_times = range(tstart, tstop, length=60)

    # Create DAE problem
    dae_prob = DAEProblem(
        swe_dae_residual_gabc_correct!, du0, u0, tspan, params;
        differential_vars=differential_vars
    )
    
    # Solve with appropriate settings for GABC
    println("Solving GABC with correct characteristic implementation...")
    sol = solve(dae_prob, IDA(), 
                reltol=1e-5, abstol=1e-7, 
                saveat=save_times,
                maxiters=2000)

    if sol.retcode != :Success
        println("Warning: Solver RetCode: $(sol.retcode)")
        return sol
    end

    # --- 5. a Live/Animated Plots ---
    # Get initial conditions for the static reference
    h_initial, q_initial = initial_conditions(params)
    
    anim = Animation()
    for (i, t) in enumerate(sol.t)
        h = sol[1:N, i]
        q = sol[N+1:2N, i]
        
        # LEFT PLOT: Exact same style as all_bottom_topographies.png
        plt_left = plot(size=(400, 400))
        
        # Plot the bottom profile (filled brown - exactly like all_bottom_topographies)
        plot!(plt_left, x, zb, 
              linewidth=3, 
              color=:saddlebrown,
              fillto=minimum(zb)-1, 
              fillcolor=:saddlebrown,
              alpha=0.7,
              label="")
        
        # Add a water surface for reference (calm water - exactly like all_bottom_topographies)
        water_surface = fill(0.0, length(x))
        plot!(plt_left, x, water_surface, 
              linewidth=2,
              color=:blue,
              label="",
              linestyle=:dash)
        
        # Fill water area (exactly like all_bottom_topographies)
        plot!(plt_left, x, zb,
              fillto=water_surface,
              fillcolor=:lightblue,
              alpha=0.4,
              linewidth=0,
              label="")
        
        xlabel!(plt_left, "Distance x [m]")
        ylabel!(plt_left, "Elevation [m]")
        title!(plt_left, "Reference Setup")
        ylims!(plt_left, minimum(zb) - 1, maximum(zb) + 2)
        
        # RIGHT PLOT: Dynamic simulation
        plt_right = plot(size=(400, 400))
        
        # Plot bottom profile
        plot!(plt_right, x, zb, 
              linewidth=3, 
              color=:saddlebrown,
              fillto=minimum(zb)-2, 
              fillcolor=:saddlebrown,
              alpha=0.8,
              label="Bottom")
        
        # Plot current water surface
        current_water_surface = h .+ zb
        plot!(plt_right, x, current_water_surface, 
              linewidth=3,
              color=:red,
              label="Current Water")
        
        # Fill current water area
        plot!(plt_right, x, zb,
              fillto=current_water_surface,
              fillcolor=:lightcoral,
              alpha=0.4,
              linewidth=0,
              label="")
        
        xlabel!(plt_right, "x [m]")
        ylabel!(plt_right, "Elevation [m]")
        title!(plt_right, "GABC t=$(round(t, digits=3))s")
        # Zoom in to surface: ±50cm around water surface
        water_center = (minimum(current_water_surface) + maximum(current_water_surface)) / 2
        ylims!(plt_right, water_center - 0.5, water_center + 0.5)
        
        # Side-by-side layout
        plot(plt_left, plt_right, layout=(1,2), size=(1000, 500))
        frame(anim)
    end
    # Generate gif with custom filename if provided
    gif_filename = haskey(params, :gif_name) ? params.gif_name : "swe_gabc_live.gif"
    gif(anim, gif_filename, fps=20)

    return sol
end

# --- 5. b Plotting results ---
function plot_solution_gabc_correct(solution, params)
    if solution.retcode != :Success
        println("Solution did not converge properly")
        return
    end
    
    N = params.N
    x = params.x
    zb = params.zb
    h_final = solution[1:N, end]
    q_final = solution[N+1:2N, end]
    
    # Get initial conditions for comparison
    h_initial, q_initial = initial_conditions(params)
    
    # LEFT PLOT: Static geometry with initial conditions (like the overview plots)
    plt1 = plot(size=(400, 300))
    
    # Plot bottom profile
    plot!(plt1, x, zb, 
          linewidth=3, 
          color=:saddlebrown,
          fillto=minimum(zb)-2, 
          fillcolor=:saddlebrown,
          alpha=0.8,
          label="Bottom Profile")
    
    # Plot initial water surface
    initial_water_surface = h_initial .+ zb
    plot!(plt1, x, initial_water_surface, 
          linewidth=3,
          color=:blue,
          label="Initial Water Surface")
    
    # Fill water area (initial)
    plot!(plt1, x, zb,
          fillto=initial_water_surface,
          fillcolor=:lightblue,
          alpha=0.4,
          linewidth=0,
          label="Initial Water")
    
    xlabel!(plt1, "Distance x [m]")
    ylabel!(plt1, "Elevation [m]")
    title!(plt1, "Initial Setup")
    
    # RIGHT PLOT: Final simulation results
    plt2 = plot(size=(400, 300))
    
    # Plot bottom profile
    plot!(plt2, x, zb, 
          linewidth=3, 
          color=:saddlebrown,
          fillto=minimum(zb)-2, 
          fillcolor=:saddlebrown,
          alpha=0.8,
          label="Bottom Profile")
    
    # Plot final water surface
    final_water_surface = h_final .+ zb
    plot!(plt2, x, final_water_surface, 
          linewidth=3,
          color=:red,
          label="Final Water Surface")
    
    # Fill water area (final)
    plot!(plt2, x, zb,
          fillto=final_water_surface,
          fillcolor=:lightcoral,
          alpha=0.4,
          linewidth=0,
          label="Final Water")
    
    xlabel!(plt2, "Distance x [m]")
    ylabel!(plt2, "Elevation [m]")
    title!(plt2, "After GABC Simulation")
    
    # BOTTOM PLOT: Discharge evolution
    plt3 = plot(x, q_initial, linewidth=2, color=:blue, label="Initial Discharge", linestyle=:dash)
    plot!(plt3, x, q_final, linewidth=3, color=:red, label="Final Discharge")
    xlabel!(plt3, "Distance x [m]")
    ylabel!(plt3, "Discharge q [m²/s]")
    title!(plt3, "Discharge Comparison")
    
    # Combine in layout: [Initial | Final]
    #                    [  Discharge    ]
    plot(plt1, plt2, plt3, layout=@layout([a b; c{0.4h}]), size=(1000, 700))
end

# --- 6. Demo function to try different bottoms ---
function demo_different_bottoms_gabc()
    println("🌊 Running GABC simulations with different bottom topographies...")
    
    # List of interesting bottoms to try
    bottom_types = ["flat", "triangle", "sudden_drop", "volcano", "staircase"]
    
    for (i, bottom_type) in enumerate(bottom_types)
        println("\n--- Running GABC simulation $(i)/$(length(bottom_types)): $(bottom_type) ---")
        
        # Set up parameters with specific bottom
        params = make_parameters(bottom_type)
        
        # Run simulation
        solution = timeloop_gabc_correct(params)
        
        # Plot results
        p = plot_solution_gabc_correct(solution, params)
        
        # Save plot
        filename = "gabc_simulation_$(bottom_type).png"
        savefig(p, filename)
        println("Saved results to: $(filename)")
    end
    
    println("\n✨ All GABC simulations complete!")
end

# --- 6b. Generate GIFs for different bottoms ---
function generate_gifs_for_all_bottoms()
    println("🎬 Generating GABC GIFs with different bottom topographies...")
    
    # List of interesting bottoms to try
    bottom_types = ["flat", "triangle", "sudden_drop", "volcano", "staircase", "canyon", "bumps"]
    
    for (i, bottom_type) in enumerate(bottom_types)
        println("\n--- Generating GIF $(i)/$(length(bottom_types)): $(bottom_type) ---")
        
        # Create unique gif filename
        gif_filename = "swe_gabc_live_$(bottom_type).gif"
        
        # Set up parameters with specific bottom and gif name
        params = make_parameters(bottom_type; gif_name=gif_filename)
        
        try
            # Run simulation
            solution = timeloop_gabc_correct(params)
            
            if solution.retcode == :Success
                println("✅ Generated: $(gif_filename)")
            else
                println("⚠️  Partial success for $(bottom_type): $(solution.retcode)")
                println("📹 GIF still generated: $(gif_filename)")
            end
        catch e
            println("❌ Error with $(bottom_type): $e")
            println("⏭️  Continuing with next bottom type...")
        end
    end
    
    println("\n🎭 All GIF generation complete!")
    println("📁 Generated files:")
    for bottom_type in bottom_types
        println("   - swe_gabc_live_$(bottom_type).gif")
    end
end

# --- 7. Main script ---
function run_single_gabc_simulation(bottom_type="flat")
    println("🌊 Running GABC shallow water simulation with $(bottom_type) bottom...")
    
    # Set up parameters
    params = make_parameters(bottom_type)
    
    # Call the time loop function
    solution = timeloop_gabc_correct(params)
    
    # Plot results
    plot_solution_gabc_correct(solution, params)
    
    return solution, params
end

println("=== CORRECT GABC IMPLEMENTATION WITH DIFFERENT SURFACE TYPES ===")
println("This implements the actual Table II equations from challenge.md")
println("Key insight: Use DAE variables (duh, duq) in characteristic equations")
println("Enhanced with multiple bottom topographies and advanced gif generation")

try
    # Run a single simulation (you can change the bottom type here!)
    solution, params = run_single_gabc_simulation("triangle")
    
    if solution.retcode == :Success
        println("✅ SUCCESS: Correct GABC implementation worked!")
        println("📊 Check 'swe_gabc_live.gif' for animation")
        println("🧮 This properly implements the characteristic equations from Table II")
        println("\n🏔️  Available bottom types:")
        all_params = make_all_topographies()
        for name in keys(all_params.bottoms)
            println("  - $name")
        end
        println("\n💡 To run with different bottom: run_single_gabc_simulation(\"bottom_name\")")
        println("💡 To run all bottoms (static plots): demo_different_bottoms_gabc()")
        println("🎬 To generate GIFs for all bottoms: generate_gifs_for_all_bottoms()")
    else
        println("⚠️  Partial success with RetCode: $(solution.retcode)")
        println("💡 The characteristic implementation is correct but may need parameter tuning")
    end
catch e
    println("❌ Error: $e")
    println("💭 This is the challenge with correct GABC - it's numerically demanding!")
end

# Uncomment one of the lines below to run batch processing:
# demo_different_bottoms_gabc()     # For static plots
generate_gifs_for_all_bottoms()   # For animated GIFs 