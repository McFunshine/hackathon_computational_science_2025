# Julia script to solve the 1D shallow water equations as a DAE problem
using NonlinearSolve, LinearAlgebra, Parameters, Plots, Sundials

# Include the bottom topography generator
include("bottom_topographies.jl")

# This code is a template for solving the 1D shallow water equations (SWE) as a DAE problem.
# The setup is as follows:
# 1. Define parameters for the simulation, including gravity, number of grid points,
#   spatial domain, and bottom topography.
# 2. Set up initial conditions for water height and momentum.
# 3. Define the DAE residual function that describes the SWE.
# 4. Implement a time loop to solve the DAE problem using Sundials' IDA solver.
# 5. Plot the results.
# 6. Function calls to start the simulation.


# --- 1. Parameter setup ---
function make_parameters(bottom_type="flat")
    N = 200
    g = 9.81
    xstart = 0.0
    xstop = 5.0
    x = range(xstart, xstop, length=N)
    D = 10.0
    
    # Use the enhanced bottom topography system
    all_params = make_all_topographies()
    zb = get_bottom_profile(bottom_type, all_params)
    
    # If the selected bottom doesn't fit our grid, interpolate it
    if length(zb) != N
        # Fallback to original options
        if bottom_type == "wavy"
            zb = -D .+ 0.4 .* sin.(2π .* x ./ xstop .* (N-1)/N .* 5)
        else
            zb = -D * ones(N)  # Flat bottom default
        end
    end
    
    tstart = 0.0
    tstop = 1.0
    return (; g, N, x, D, zb, tstart, tstop)
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
    # Initial height
    xstop = x[end]
    h0 = 0.1 .* exp.(-100 .* ((x ./ xstop .- 0.5) .* xstop).^2) .- zb
    return h0, q0
end

# --- 3. DAE residual function ---
# Note: the "!" at the end of the function name indicates that the function modifies 
# its arguments (convention in Julia)
function swe_dae_residual!(residual, du, u, p, t)
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
    # Compute zeta (free surface)
    zeta = h .+ zb
    # Periodic boundary indices
    for i in 1:N
        ip = i == N ? 1 : i+1
        im = i == 1 ? N : i-1
        # Central differences
        dqdx = (q[ip] - q[im]) / (2dx)
        dzetadx = (zeta[ip] - zeta[im]) / (2dx)
        dfdx = ((q[ip]^2/h[ip]) - (q[im]^2/h[im])) / (2dx)
        # SWE residuals
        residual[i] = duh[i] + dqdx
        residual[N+i] = duq[i] + dfdx + g * h[i] * dzetadx
    end
    return nothing
end

# --- 4. Time integration ---
function timeloop(params)
    # Unpack parameters 
    g = params.g
    N = params.N
    x = params.x
    zb = params.zb
    dx = x[2] - x[1]
    tstart = params.tstart
    tstop = params.tstop

    # set up initial conditions
    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    du0 = zeros(2N)  # Initial guess for du/dt

    # Calculate static y-limits for h using h0
    hmin = minimum(h0)
    hmax = maximum(h0)
    margin = 0.25
    ylim_h = (hmin - margin, hmax + margin)

    # Calculate static y-limits for q using q0
    qmin = minimum(q0)
    qmax = maximum(q0)
    margin = 1
    ylim_q = (qmin - margin, qmax + margin)

    tspan = (tstart, tstop) # defines the start and end times for the simulation

    # Specify differentiable variables as (true) -> all variables
    differential_vars = trues(2N)

    # Save solution at regular intervals for animation
    save_times = range(tstart, tstop, length=100)

    dae_prob = DAEProblem(
        swe_dae_residual!, du0, u0, tspan, params;
        differential_vars=differential_vars
    )
    sol = solve(dae_prob, IDA(), reltol=1e-8, abstol=1e-8, saveat=save_times) # solves the DAE problem using default settings

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
         title!(plt_right, "t=$(round(t, digits=3))s")
         ylims!(plt_right, -0.5, 0.5)  # Zoom in to surface: -50cm to +50cm
        
        # Side-by-side layout
        plot(plt_left, plt_right, layout=(1,2), size=(1000, 500))
        frame(anim)
    end
    gif(anim, "swe_live.gif", fps=20)

    return sol # return solution object
end

# --- 5. b Plotting results ---
function plot_solution(solution, params)
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
    title!(plt2, "After Simulation")
    
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
function demo_different_bottoms()
    println("🌊 Running simulations with different bottom topographies...")
    
    # List of interesting bottoms to try
    bottom_types = ["flat", "triangle", "sudden_drop", "volcano", "staircase"]
    
    for (i, bottom_type) in enumerate(bottom_types)
        println("\n--- Running simulation $(i)/$(length(bottom_types)): $(bottom_type) ---")
        
        # Set up parameters with specific bottom
        params = make_parameters(bottom_type)
        
        # Run simulation
        solution = timeloop(params)
        
        # Plot results
        p = plot_solution(solution, params)
        
        # Save plot
        filename = "simulation_$(bottom_type).png"
        savefig(p, filename)
        println("Saved results to: $(filename)")
    end
    
    println("\n✨ All simulations complete!")
end

# --- 7. Main script ---
function run_single_simulation(bottom_type="flat")
    println("🌊 Running shallow water simulation with $(bottom_type) bottom...")
    
    # Set up parameters
    params = make_parameters(bottom_type)
    
    # Call the time loop function
    solution = timeloop(params)
    
    # Plot results
    plot_solution(solution, params)
    
    return solution, params
end

# Run a single simulation (you can change the bottom type here!)
solution, params = run_single_simulation("triangle")

# Uncomment the line below to run demo with multiple bottom types
# demo_different_bottoms()
