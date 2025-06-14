# Julia script to solve the 1D shallow water equations as a DAE problem
using NonlinearSolve, LinearAlgebra, Parameters, Plots, Sundials

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
function make_parameters()
    # From Table I in the document
    g = 9.81          # gravitational acceleration
    N = 200           # grid points (nx from table)
    xmax = 5.0        # domain length
    x = range(0.0, xmax, length=N)  # spatial grid
    D = 10.0          # domain depth
    
    # Bottom topography (two options from table)
    zb_flat = fill(-D, N)  # flat bottom: -D
    zb_wavy = -D .+ 0.4 .* sin.(2π .* x ./ xmax .* (N-1)/N .* 5)  # wavy bottom
    
    tstart = 0.0      # initial time
    tstop = 1.0       # final time
    
    # Choose which bottom to use
    zb = zb_flat  # or zb_wavy
    
    return (; g, N, x, D, zb, tstart, tstop, xmax)
end

# --- 2. Initial condition ---
function initial_conditions(params)
    @unpack N, x, xmax, D, zb = params
    
    # Initial discharge (zero everywhere)
    q = zeros(N)
    
    # Initial height - Gaussian disturbance
    h = 0.1 .* exp.(-100 .* ((x ./ xmax .- 0.5) ./ xmax).^2) .- zb
    
    return h, q
end

# --- 3. DAE residual function ---
# Note: the "!" at the end of the function name indicates that the function modifies 
# its arguments (convention in Julia)
function swe_dae_residual!(residual, du, u, p, t)
    @unpack g, N, x, zb = p
    
    # Extract h and q from the state vector u
    # u = [h1, h2, ..., hN, q1, q2, ..., qN]
    h = u[1:N]
    q = u[N+1:2N]
    
    # Extract time derivatives from du
    dhdt = du[1:N]
    dqdt = du[N+1:2N]
    
    # Compute grid spacing (assuming uniform grid)
    dx = x[2] - x[1]
    
    # Initialize residuals
    # First N equations: continuity equation
    # Next N equations: momentum equation
    
    # Continuity equation: ∂h/∂t + ∂q/∂x = 0
    # Rearranged: ∂h/∂t + ∂q/∂x = 0
    for i in 1:N
        if i == 1
            # Forward difference at left boundary
            dqdx = (q[i+1] - q[i]) / dx
        elseif i == N
            # Backward difference at right boundary
            dqdx = (q[i] - q[i-1]) / dx
        else
            # Central difference for interior points
            dqdx = (q[i+1] - q[i-1]) / (2*dx)
        end
        
        residual[i] = dhdt[i] + dqdx
    end
    
    # Momentum equation: ∂q/∂t + ∂(q²/h)/∂x = -gh∂ζ/∂x
    # where ζ = h + zb (free surface level)
    # Simplifying: ∂q/∂t + ∂(q²/h)/∂x + gh∂(h+zb)/∂x = 0
    for i in 1:N
        # Compute ∂(q²/h)/∂x
        if i == 1
            # Forward difference at left boundary
            flux_left = h[i] > 1e-10 ? q[i]^2 / h[i] : 0.0
            flux_right = h[i+1] > 1e-10 ? q[i+1]^2 / h[i+1] : 0.0
            dflux_dx = (flux_right - flux_left) / dx
        elseif i == N
            # Backward difference at right boundary
            flux_left = h[i-1] > 1e-10 ? q[i-1]^2 / h[i-1] : 0.0
            flux_right = h[i] > 1e-10 ? q[i]^2 / h[i] : 0.0
            dflux_dx = (flux_right - flux_left) / dx
        else
            # Central difference for interior points
            flux_left = h[i-1] > 1e-10 ? q[i-1]^2 / h[i-1] : 0.0
            flux_right = h[i+1] > 1e-10 ? q[i+1]^2 / h[i+1] : 0.0
            dflux_dx = (flux_right - flux_left) / (2*dx)
        end
        
        # Compute ∂ζ/∂x = ∂(h + zb)/∂x
        if i == 1
            # Forward difference
            dzeta_dx = ((h[i+1] + zb[i+1]) - (h[i] + zb[i])) / dx
        elseif i == N
            # Backward difference
            dzeta_dx = ((h[i] + zb[i]) - (h[i-1] + zb[i-1])) / dx
        else
            # Central difference
            dzeta_dx = ((h[i+1] + zb[i+1]) - (h[i-1] + zb[i-1])) / (2*dx)
        end
        
        # Momentum equation residual
        # Note: We're omitting the friction term c_f*q|q|/h² for simplicity
        residual[N+i] = dqdt[i] + dflux_dx + g * h[i] * dzeta_dx
    end
    
    return nothing
end

# --- 4. Time integration ---
function timeloop(params)
    # Unpack parameters 
    @unpack g, N, x, D, zb, tstart, tstop, xmax = params

    # set up initial conditions
    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    du0 = zeros(2N)  # Initial guess for du/dt

    tspan = (tstart, tstop) # defines the start and end times for the simulation

    # Specify differentiable variables as (true) -> all variables
    differential_vars = trues(2N)

    dae_prob = DAEProblem(
        swe_dae_residual!, du0, u0, tspan, params;
        differential_vars=differential_vars
    )
    sol = solve(dae_prob, IDA(), reltol=1e-8, abstol=1e-8) # solves the DAE problem using default settings

    # --- 5. a Live Plots ---

    return sol # return solution object
end

# --- 5. b Plotting results ---
function plot_results(sol, params)
    @unpack N, x, zb, g = params
    
    # Extract solution at different time points
    t_plot = range(sol.t[1], sol.t[end], length=min(50, length(sol.t)))
    
    # Create subplots
    p1 = plot(layout=(2, 2), size=(1000, 800))
    
    # Plot 1: Water height h(x,t) evolution
    for (i, t) in enumerate(t_plot[1:5:end])  # Plot every 5th time step
        u_t = sol(t)
        h_t = u_t[1:N]
        plot!(p1, x, h_t, subplot=1, 
              label=i==1 ? "h(x,t)" : "", 
              alpha=0.7, 
              title="Water Height Evolution")
    end
    xlabel!(p1, "x [m]", subplot=1)
    ylabel!(p1, "h [m]", subplot=1)
    
    # Plot 2: Discharge q(x,t) evolution  
    for (i, t) in enumerate(t_plot[1:5:end])  # Plot every 5th time step
        u_t = sol(t)
        q_t = u_t[N+1:2N]
        plot!(p1, x, q_t, subplot=2,
              label=i==1 ? "q(x,t)" : "",
              alpha=0.7,
              title="Discharge Evolution")
    end
    xlabel!(p1, "x [m]", subplot=2)
    ylabel!(p1, "q [m²/s]", subplot=2)
    
    # Plot 3: Free surface elevation ζ = h + zb at final time
    u_final = sol(sol.t[end])
    h_final = u_final[1:N]
    zeta_final = h_final .+ zb
    
    plot!(p1, x, zb, subplot=3, label="Bottom zb", linewidth=2, color=:brown)
    plot!(p1, x, zeta_final, subplot=3, label="Free surface ζ", linewidth=2, color=:blue)
    fill_between!(p1, x, zb, zeta_final, subplot=3, alpha=0.3, color=:lightblue, label="Water")
    xlabel!(p1, "x [m]", subplot=3)
    ylabel!(p1, "Elevation [m]", subplot=3)
    title!(p1, "Final Water Profile", subplot=3)
    
    # Plot 4: Time series at a specific location (middle of domain)
    i_mid = div(N, 2)
    t_series = sol.t
    h_series = [sol(t)[i_mid] for t in t_series]
    q_series = [sol(t)[N + i_mid] for t in t_series]
    
    plot!(p1, t_series, h_series, subplot=4, label="h at x=$(round(x[i_mid], digits=2))", linewidth=2)
    plot!(p1, t_series, q_series, subplot=4, label="q at x=$(round(x[i_mid], digits=2))", linewidth=2)
    xlabel!(p1, "Time [s]", subplot=4)
    ylabel!(p1, "Value", subplot=4)
    title!(p1, "Time Series at Mid-Domain", subplot=4)
    
    # Save plots to files instead of displaying
    savefig(p1, "shallow_water_analysis.png")
    println("Saved comprehensive analysis plot to: shallow_water_analysis.png")
    
    # Additional plot: Animation-style plot of final state
    p2 = plot(size=(800, 400))
    plot!(p2, x, zb, label="Bottom", linewidth=2, color=:brown, fillto=minimum(zb)-0.5, fillcolor=:saddlebrown)
    plot!(p2, x, zeta_final, label="Free Surface", linewidth=3, color=:blue)
    fill_between!(p2, x, zb, zeta_final, alpha=0.4, color=:lightblue, label="Water")
    
    xlabel!(p2, "Distance x [m]")
    ylabel!(p2, "Elevation [m]")
    title!(p2, "Final Water Profile (t = $(round(sol.t[end], digits=2)) s)")
    
    # Save the second plot too
    savefig(p2, "final_water_profile.png")
    println("Saved final water profile plot to: final_water_profile.png")
    
    return p1, p2
end

# Helper function for fill_between (if not available in Plots.jl)
function fill_between!(p, x, y1, y2; subplot=1, alpha=0.3, color=:lightblue, label="")
    # Create filled area between y1 and y2
    x_fill = vcat(x, reverse(x))
    y_fill = vcat(y1, reverse(y2))
    plot!(p, x_fill, y_fill, seriestype=:shape, subplot=subplot, 
          alpha=alpha, color=color, label=label, linewidth=0)
end

# --- 6. Main script ---
# Set up parameters
params = make_parameters()
# Call the time loop function
solution = timeloop(params)

# Plot the results
plot_results(solution, params)