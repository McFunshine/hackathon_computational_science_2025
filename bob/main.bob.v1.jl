# 1D Shallow Water Equation Solver using SciML
# Place this in /home/hackonaut/challenge/main.bob.jl (or similar)

using NonlinearSolve, LinearAlgebra, Printf

# === Problem Parameters ===
const g = 9.81          # gravity [m/s^2]
const D = 10.0          # domain depth [m]
const nx = 200          # number of grid points
const L = 5.0           # domain length [m]
const tstart = 0.0      # initial time [s]
const tstop = 1.0       # final time [s]
const dt = 1e-3         # time step [s]
const cf = 0.0          # friction coefficient (set to 0 for now)

x = LinRange(0, L, nx)             # spatial grid
dx = x[2] - x[1]

# === Bed Profile ===
zb_flat = -D * ones(nx)
zb_wavy = -D .+ 0.4 .* sin.(2π .* x ./ L .* (nx-1) ./ (nx*5))
zb = zb_flat      # or use zb_wavy for the wavy case

# === Initial Conditions ===
h0 = 0.1 .* exp.(-100 .* ((x ./ L .- 0.5) .* L).^2) .- zb
q0 = zeros(nx)

# === Pack state as single vector ===
function pack(h, q)
    return vcat(h, q)
end

function unpack(U)
    n = length(U) ÷ 2
    h = U[1:n]
    q = U[n+1:end]
    return h, q
end

U0 = pack(h0, q0)

params = (; g, nx, x, dx, D, zb, cf)

# === Residual Function (to be used by SciML) ===
function shallow_water_residual!(res, U, p)
    nx, dx, g, zb, cf = p.nx, p.dx, p.g, p.zb, p.cf
    h, q = unpack(U)
    T = eltype(U)  # <-- Use correct type for AD compatibility!

    # Use the correct type for all arrays!
    dhdx = zeros(T, nx)
    dqdx = zeros(T, nx)
    dq2hdx = zeros(T, nx)
    dzetadx = zeros(T, nx)
    q_abs = abs.(q)
    
    # Periodic indices for BC
    left(i)  = i == 1    ? nx : i-1
    right(i) = i == nx   ? 1  : i+1
    
    # Derivatives
    for i in 1:nx
        dhdx[i] = (h[right(i)] - h[left(i)]) / (2dx)
        dqdx[i] = (q[right(i)] - q[left(i)]) / (2dx)
        dq2hdx[i] = ((q[right(i)]^2 / h[right(i)]) - (q[left(i)]^2 / h[left(i)])) / (2dx)
        dzetadx[i] = ((h[right(i)] + zb[right(i)]) - (h[left(i)] + zb[left(i)])) / (2dx)
    end

    # Residuals (from discretized PDEs)
    res[1:nx] = dqdx
    res[nx+1:2nx] = dq2hdx .+ g .* h .* dzetadx .+ cf .* q .* q_abs ./ (h.^2)
    
    return res
end

# === Time Integration (Backward Euler, via NonlinearSolve) ===
function timeloop(U0, params; dt=1e-3, tstart=0.0, tstop=1.0)
    nsteps = Int(round((tstop-tstart)/dt))
    U_hist = [U0]
    t_hist = [tstart]

    U_prev = copy(U0)
    t = tstart

    for step = 1:nsteps
        t += dt
        # Define nonlinear problem at this step
        prob = NonlinearProblem(
            (res, U, p) -> begin
                shallow_water_residual!(res, U, p)
                res .-= (U .- U_prev) ./ dt
            end,
            U_prev, params
        )
        sol = solve(prob, NewtonRaphson(), abstol=1e-8, reltol=1e-8, maxiters=50)
        push!(U_hist, sol.u)
        push!(t_hist, t)
        U_prev = sol.u
        @printf("Step %d / %d, t=%.3f\n", step, nsteps, t)
    end

    return U_hist, t_hist
end

# === Run Simulation ===
U_hist, t_hist = timeloop(U0, params; dt=dt, tstart=tstart, tstop=tstop)

# === Example: Save results for plotting (to .csv or .jld2 as needed) ===
# Write your save/plot code here

