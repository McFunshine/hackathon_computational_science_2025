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
    N = 200
    g = 9.81
    xstart = 0.0
    xstop = 5.0
    x = range(xstart, xstop, length=N)
    D = 10.0
    # Flat bottom
    zb = -D * ones(N)
    # Wavy bottom
    #zb = -D .+ 0.4 .* sin.(2pi .* x ./ xstop .* (N-1)/N .* 5)
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
    anim = Animation()
    for (i, t) in enumerate(sol.t)
        h = sol[1:N, i]
        q = sol[N+1:2N, i]
        plt1 = plot(x, h, xlabel="x", ylabel="Water Height h", title="Water Height at t=$(round(t, digits=3))", legend=false, ylim=(minimum(h0), maximum(h0)+params.D))
        plt2 = plot(x, q, xlabel="x", ylabel="Discharge q", title="Discharge at t=$(round(t, digits=3))", legend=false, ylim=(minimum(q0)-1, maximum(q0)+1))
        plot(plt1, plt2, layout=(2,1))
        frame(anim)
    end
    gif(anim, "swe_live.gif", fps=20)

    return sol # return solution object
end

# --- 5. b Plotting results ---
function plot_solution(solution, params)
    N = params.N
    x = params.x
    h = solution[1:N, end]
    q = solution[N+1:2N, end]
    plt1 = plot(x, h, xlabel="x", ylabel="Water Height h", title="Final Water Height", legend=false)
    plt2 = plot(x, q, xlabel="x", ylabel="Discharge q", title="Final Discharge", legend=false)
    plot(plt1, plt2, layout=(2,1))
end

# --- 6. Main script ---
# Set up parameters
params = make_parameters()
# Call the time loop function
solution = timeloop(params)
plot_solution(solution, params)
