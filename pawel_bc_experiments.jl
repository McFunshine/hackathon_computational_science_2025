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
    # Friction coefficient
    cf = 0.01
    # Example: Dirichlet/Neumann BCs
    bc_type = :dirichlet_neumann
    # Prescribe q at left (can be a function of t)
    qin(t) = t < 0.2 ? 0.2 : 0.0
    # Prescribe zeta at right (can be a function of t)
    zetaout(t) = D
    bc_values = Dict(:qin => qin, :zetaout => zetaout)
    return (; g, N, x, D, zb, tstart, tstop, cf, bc_type, bc_values)
end

# --- 2. Initial condition ---
function initial_conditions(params)
    g = params.g
    N = params.N
    x = params.x
    D = params.D
    zb = params.zb
    bc_type = get(params, :bc_type, :periodic)
    bc_values = get(params, :bc_values, Dict())
    
    # Initial discharge
    q0 = zeros(N)
    # Initial height
    xstop = x[end]
    h0 = 0.1 .* exp.(-100 .* ((x ./ xstop .- 0.5) .* xstop).^2) .- zb
    
    # Make initial conditions consistent with boundary conditions
    if bc_type == :dirichlet_neumann
        # Set initial discharge at left boundary to match qin(0)
        q0[1] = bc_values[:qin](0.0)
        # Set initial height at right boundary to match zetaout(0)
        h0[N] = bc_values[:zetaout](0.0) - zb[N]
        println("Initial conditions adjusted for BC consistency:")
        println("  q0[1] = $(q0[1]) (should match qin(0) = $(bc_values[:qin](0.0)))")
        println("  h0[N] + zb[N] = $(h0[N] + zb[N]) (should match zetaout(0) = $(bc_values[:zetaout](0.0)))")
    end
    
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
    cf = p.cf
    dx = x[2] - x[1]
    bc_type = get(p, :bc_type, :periodic)
    bc_values = get(p, :bc_values, Dict())
    # Unpack state
    h = @view u[1:N]
    q = @view u[N+1:2N]
    duh = @view du[1:N]
    duq = @view du[N+1:2N]
    # Compute zeta (free surface)
    zeta = h .+ zb
    # --- Left boundary (i=1) ---
    if bc_type == :dirichlet_neumann
        # Dirichlet for q (prescribed discharge)
        qin = bc_values[:qin](t)
        residual[N+1] = q[1] - qin
        # For h, use one-sided difference for dqdx
        dqdx = (q[2] - q[1]) / dx
        residual[1] = duh[1] + dqdx
    else
        # Periodic (default)
        im = N
        ip = 2
        dqdx = (q[ip] - q[im]) / (2dx)
        dzetadx = (zeta[ip] - zeta[im]) / (2dx)
        dfdx = ((q[ip]^2/h[ip]) - (q[im]^2/h[im])) / (2dx)
        friction = cf * q[1] * abs(q[1]) / h[1]^2
        residual[1] = duh[1] + dqdx
        residual[N+1] = duq[1] + dfdx + g * h[1] * dzetadx + friction
    end
    # --- Interior points (i=2:N-1) ---
    for i in 2:N-1
        ip = i + 1
        im = i - 1
        dqdx = (q[ip] - q[im]) / (2dx)
        dzetadx = (zeta[ip] - zeta[im]) / (2dx)
        dfdx = ((q[ip]^2/h[ip]) - (q[im]^2/h[im])) / (2dx)
        friction = cf * q[i] * abs(q[i]) / h[i]^2
        residual[i] = duh[i] + dqdx
        residual[N+i] = duq[i] + dfdx + g * h[i] * dzetadx + friction
    end
    # --- Right boundary (i=N) ---
    if bc_type == :dirichlet_neumann
        # Dirichlet for zeta (prescribed water level)
        zetaout = bc_values[:zetaout](t)
        residual[N] = h[N] + zb[N] - zetaout
        # For q, use one-sided difference for dfdx
        dfdx = (q[N]^2/h[N] - q[N-1]^2/h[N-1]) / dx
        dzetadx = (zeta[N] - zeta[N-1]) / dx
        friction = cf * q[N] * abs(q[N]) / h[N]^2
        residual[2N] = duq[N] + dfdx + g * h[N] * dzetadx + friction
    else
        # Periodic (default)
        im = N-1
        ip = 1
        dqdx = (q[ip] - q[im]) / (2dx)
        dzetadx = (zeta[ip] - zeta[im]) / (2dx)
        dfdx = ((q[ip]^2/h[ip]) - (q[im]^2/h[im])) / (2dx)
        friction = cf * q[N] * abs(q[N]) / h[N]^2
        residual[N] = duh[N] + dqdx
        residual[2N] = duq[N] + dfdx + g * h[N] * dzetadx + friction
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
    anim = Animation()
    for (i, t) in enumerate(sol.t)
        h = sol[1:N, i]
        q = sol[N+1:2N, i]
        plt1 = plot(x, h, xlabel="x", ylabel="Water Height h", title="Water Height at t=$(round(t, digits=3))", legend=false, ylim=ylim_h)
        plt2 = plot(x, q, xlabel="x", ylabel="Discharge q", title="Discharge at t=$(round(t, digits=3))", legend=false, ylim=ylim_q)
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
