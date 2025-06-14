# 1D Shallow Water Equation Solver - Complete Implementation
# Computational Science NL Hackathon 2025
# Implements periodic, Dirichlet/Neumann, and GABC boundary conditions

using NonlinearSolve, LinearAlgebra, Parameters, Plots, Sundials

# --- 1. Parameter setup following challenge.md requirements ---
function make_parameters(boundary_type=:periodic; custom_params...)
    """
    Create parameters tuple as specified in challenge.md
    boundary_type: :periodic, :dirichlet_neumann, or :gabc
    """
    # Default parameters from challenge test case
    N = 200
    g = 9.81
    xstart = 0.0
    xstop = 5.0
    x = range(xstart, xstop, length=N)
    D = 10.0
    
    # Bottom topography options from challenge
    if haskey(custom_params, :flat_bottom) && custom_params[:flat_bottom]
        zb = -D * ones(N)  # Flat bottom zb,1
    else
        # Wavy bottom zb,2 from challenge specification
        zb = -D .+ 0.4 .* sin.(2π .* x ./ xstop .* (N-1)/N .* 5)
    end
    
    tstart = 0.0
    tstop = 1.0
    
    # Allow parameter customization
    for (key, value) in custom_params
        if key == :N
            N = value
            x = range(xstart, xstop, length=N)
            if haskey(custom_params, :flat_bottom) && custom_params[:flat_bottom]
                zb = -D * ones(N)
            else
                zb = -D .+ 0.4 .* sin.(2π .* x ./ xstop .* (N-1)/N .* 5)
            end
        elseif key == :tstop
            tstop = value
        elseif key == :g
            g = value
        elseif key == :D
            D = value
        end
    end
    
    return (; g, N, x, D, zb, tstart, tstop, boundary_type)
end

# --- 2. Initial conditions from challenge specification ---
function initial_conditions(params)
    """
    Initial conditions as specified in challenge.md
    """
    g = params.g
    N = params.N
    x = params.x
    D = params.D
    zb = params.zb
    
    # Initial discharge q0 = 0.0 (from challenge)
    q0 = zeros(N)
    
    # Initial height h0 from challenge specification
    xmax = x[end]
    h0 = 0.1 .* exp.(-100 .* ((x ./ xmax .- 0.5) .* xmax).^2) .- zb
    
    # Ensure positive water depth
    h0 = max.(h0, 0.01)
    
    return h0, q0
end

# --- 3. Boundary condition functions for GABC ---
function prescribed_incoming_left(t)
    """Prescribed incoming characteristic at left boundary for GABC"""
    amplitude = 0.02  # Sinusoidal signal amplitude
    frequency = 0.5   # Frequency for sinusoidal signal
    return amplitude * sin(2π * frequency * t)
end

function prescribed_incoming_right(t)
    """Prescribed incoming characteristic at right boundary for GABC"""
    return 0.0  # Absorbing boundary
end

# --- 4. Dirichlet/Neumann boundary conditions ---
function prescribed_discharge_left(t)
    """Prescribed discharge at upstream boundary (Dirichlet)"""
    amplitude = 0.1
    frequency = 0.5
    return amplitude * sin(2π * frequency * t)
end

function prescribed_water_level_right(t)
    """Prescribed water level at downstream boundary (Neumann)"""
    return 0.0  # Fixed water level
end

# --- 5. Unified DAE residual function ---
function swe_dae_residual!(residual, du, u, p, t)
    """
    Unified residual function supporting all boundary condition types
    """
    # Unpack parameters
    g = p.g
    N = p.N
    x = p.x
    zb = p.zb
    boundary_type = p.boundary_type
    dx = x[2] - x[1]
    
    # Unpack state
    h = @view u[1:N]
    q = @view u[N+1:2N]
    duh = @view du[1:N]
    duq = @view du[N+1:2N]
    
    # Safety for shallow water
    h_min = 1e-6
    h_safe = max.(h, h_min)
    
    # Compute free surface
    zeta = h_safe .+ zb
    
    # Interior points (always use standard SWE)
    start_idx = boundary_type == :periodic ? 1 : 2
    end_idx = boundary_type == :periodic ? N : N-1
    
    for i in start_idx:end_idx
        if boundary_type == :periodic
            # Periodic boundary indices
            ip = i == N ? 1 : i+1
            im = i == 1 ? N : i-1
        else
            # Standard interior points
            ip = i + 1
            im = i - 1
        end
        
        # Central differences
        dqdx = (q[ip] - q[im]) / (2*dx)
        dzetadx = (zeta[ip] - zeta[im]) / (2*dx)
        dfdx = ((q[ip]^2/h_safe[ip]) - (q[im]^2/h_safe[im])) / (2*dx)
        
        # SWE residuals
        residual[i] = duh[i] + dqdx
        residual[N+i] = duq[i] + dfdx + g * h_safe[i] * dzetadx
    end
    
    # Boundary conditions
    if boundary_type == :dirichlet_neumann
        # Left boundary (i=1) - Dirichlet (prescribed discharge)
        i = 1
        dqdx = (q[i+1] - q[i]) / dx
        dzetadx = (zeta[i+1] - zeta[i]) / dx
        dfdx = ((q[i+1]^2/h_safe[i+1]) - (q[i]^2/h_safe[i])) / dx
        
        # Continuity equation
        residual[i] = duh[i] + dqdx
        # Prescribed discharge
        residual[N+i] = q[i] - prescribed_discharge_left(t)
        
        # Right boundary (i=N) - Neumann (prescribed water level)
        i = N
        dqdx = (q[i] - q[i-1]) / dx
        dzetadx = (zeta[i] - zeta[i-1]) / dx
        dfdx = ((q[i]^2/h_safe[i]) - (q[i-1]^2/h_safe[i-1])) / dx
        
        # Prescribed water level (zeta = h + zb)
        residual[i] = zeta[i] - (prescribed_water_level_right(t) + zb[i])
        # Momentum equation
        residual[N+i] = duq[i] + dfdx + g * h_safe[i] * dzetadx
        
    elseif boundary_type == :gabc
        # Left boundary (i=1) - GABC implementation
        i = 1
        dqdx = (q[i+1] - q[i]) / dx
        c = sqrt(g * h_safe[i])
        
        # Incoming characteristic from Table II
        incoming_char = (1/h_safe[i]) * (duq[i] - (q[i]/h_safe[i])*duh[i] + c*duh[i])
        prescribed_incoming = prescribed_incoming_left(t)
        
        # Continuity equation
        residual[i] = duh[i] + dqdx
        # GABC constraint
        residual[N+i] = incoming_char - prescribed_incoming
        
        # Right boundary (i=N) - GABC implementation  
        i = N
        dqdx = (q[i] - q[i-1]) / dx
        c = sqrt(g * h_safe[i])
        
        # Incoming characteristic from Table II
        incoming_char = (1/h_safe[i]) * (duq[i] - (q[i]/h_safe[i])*duh[i] - c*duh[i])
        prescribed_incoming = prescribed_incoming_right(t)
        
        # Continuity equation
        residual[i] = duh[i] + dqdx
        # GABC constraint
        residual[N+i] = incoming_char - prescribed_incoming
    end
    
    return nothing
end

# --- 6. Main timeloop function as specified in challenge.md ---
function timeloop(params)
    """
    Main solver function as specified in challenge.md
    Input: parameters tuple (; g, N, x, D, zb, tstart, tstop, boundary_type)
    Output: solution object
    """
    # Unpack parameters 
    g = params.g
    N = params.N
    x = params.x
    zb = params.zb
    tstart = params.tstart
    tstop = params.tstop
    boundary_type = params.boundary_type
    
    println("=== 1D Shallow Water Equation Solver ===")
    println("Boundary condition type: $(boundary_type)")
    println("Grid points: $(N)")
    println("Time span: $(tstart) to $(tstop)")

    # Set up initial conditions
    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    du0 = zeros(2N)
    
    # For GABC, ensure consistent initial conditions
    if boundary_type == :gabc
        # Left boundary initial condition
        c = sqrt(g * h0[1])
        prescribed = prescribed_incoming_left(0.0)
        du0[N+1] = h0[1] * prescribed
        
        # Right boundary initial condition
        c = sqrt(g * h0[N])
        prescribed = prescribed_incoming_right(0.0)
        du0[N+N] = h0[N] * prescribed
    end
    
    # Calculate plotting limits
    zeta0 = h0 .+ zb
    zetamin = minimum(zeta0) - 0.3
    zetamax = maximum(zeta0) + 0.3
    ylim_zeta = (zetamin, zetamax)
    
    qmin = minimum(q0) - 1.0
    qmax = maximum(q0) + 1.0
    ylim_q = (qmin, qmax)

    tspan = (tstart, tstop)
    differential_vars = trues(2N)
    save_times = range(tstart, tstop, length=100)

    # Create and solve DAE problem
    dae_prob = DAEProblem(
        swe_dae_residual!, du0, u0, tspan, params;
        differential_vars=differential_vars
    )
    
    println("Solving DAE system...")
    sol = solve(dae_prob, IDA(), 
                reltol=1e-6, abstol=1e-8, 
                saveat=save_times,
                maxiters=2000)

    if sol.retcode != :Success
        println("Warning: Solver RetCode: $(sol.retcode)")
    else
        println("✅ Solution successful!")
    end

    # Create animation
    println("Creating animation...")
    anim = Animation()
    for (i, t) in enumerate(sol.t)
        h = sol[1:N, i]
        q = sol[N+1:2N, i]
        zeta = h .+ zb
        
        plt1 = plot(x, zeta, xlabel="x", ylabel="Free Surface ζ", 
                   title="$(boundary_type): Free Surface at t=$(round(t, digits=3))", 
                   legend=:topright, label="Free Surface", linewidth=2, color=:blue,
                   ylim=ylim_zeta)
        plot!(x, zb, linestyle=:dash, color=:gray, alpha=0.5, label="Bottom")
        
        plt2 = plot(x, q, xlabel="x", ylabel="Discharge q", 
                   title="$(boundary_type): Discharge at t=$(round(t, digits=3))", 
                   legend=false, linewidth=2, color=:red, ylim=ylim_q)
        
        plot(plt1, plt2, layout=(2,1))
        frame(anim)
    end
    
    filename = "swe_$(boundary_type).gif"
    gif(anim, filename, fps=20)
    println("📹 Animation saved as: $(filename)")

    return sol
end

# --- 7. Final plotting function ---
function plot_solution(solution, params)
    """Plot final solution state"""
    if solution.retcode != :Success
        println("Solution did not converge properly")
        return
    end
    
    N = params.N
    x = params.x
    zb = params.zb
    boundary_type = params.boundary_type
    
    h = solution[1:N, end]
    q = solution[N+1:2N, end]
    zeta = h .+ zb
    
    plt1 = plot(x, zeta, xlabel="x", ylabel="Free Surface ζ", 
               title="$(boundary_type): Final Free Surface", legend=:topright, 
               label="Free Surface", linewidth=2, color=:blue)
    plot!(x, zb, linestyle=:dash, color=:gray, alpha=0.7, label="Bottom")
    
    plt2 = plot(x, q, xlabel="x", ylabel="Discharge q", 
               title="$(boundary_type): Final Discharge", legend=false, 
               linewidth=2, color=:red)
    
    plot(plt1, plt2, layout=(2,1), size=(800, 600))
end

# --- 8. Demo and test functions ---
function run_all_tests()
    """Run all three boundary condition types for comparison"""
    
    println("\n🔄 Testing all boundary condition implementations...\n")
    
    # Test 1: Periodic boundary conditions
    println("1️⃣  PERIODIC BOUNDARY CONDITIONS")
    params_periodic = make_parameters(:periodic, tstop=1.0)
    sol_periodic = timeloop(params_periodic)
    plot_solution(sol_periodic, params_periodic)
    
    # Test 2: Dirichlet/Neumann boundary conditions
    println("\n2️⃣  DIRICHLET/NEUMANN BOUNDARY CONDITIONS")
    params_dn = make_parameters(:dirichlet_neumann, tstop=1.0, flat_bottom=true)
    sol_dn = timeloop(params_dn)
    plot_solution(sol_dn, params_dn)
    
    # Test 3: GABC boundary conditions (BONUS)
    println("\n3️⃣  GABC BOUNDARY CONDITIONS (BONUS)")
    params_gabc = make_parameters(:gabc, tstop=0.8, flat_bottom=true, N=100)
    sol_gabc = timeloop(params_gabc)
    plot_solution(sol_gabc, params_gabc)
    
    println("\n✅ All tests completed!")
    println("Check the generated GIF files for animations of each boundary condition type.")
end

# --- 9. Main entry point following challenge requirements ---
if abspath(PROGRAM_FILE) == @__FILE__
    println("🌊 1D Shallow Water Equation Solver")
    println("Computational Science NL Hackathon 2025")
    println("========================================")
    
    # Default run with parameters as specified in challenge.md
    params = make_parameters(:periodic)  # Start with periodic as baseline
    solution = timeloop(params)
    plot_solution(solution, params)
    
    # Uncomment to run all tests
    run_all_tests()
end 