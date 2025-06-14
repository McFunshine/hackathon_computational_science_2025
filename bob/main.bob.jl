# 1D Shallow Water Equation Solver using SciML
# Author: [Your Name]
# Purpose: Solves the 1D Shallow Water Equations with multiple bed profile options,
# including a real Rhine cross-section loaded from a CSV file.

using NonlinearSolve
using LinearAlgebra
using Printf
using Plots
using DelimitedFiles
using Interpolations

# === Problem Parameters (EDITABLE) ===
const g = 9.81            # Gravitational acceleration [m/s^2]
const D = 10.0            # Reference depth [m]
const nx = 200            # Number of grid points in the domain
const L = 5.0             # Domain length [km]  # [You may adjust units]
const tstart = 0.0        # Initial time [s]
const tstop = 1.0         # Final time [s]
const dt = 1e-3           # Time step [s]
const cf = 0.0            # Bed friction coefficient [-]

x = LinRange(0, L, nx)
dx = x[2] - x[1]

# === Bed Profile Definitions ===
zb_flat = -D * ones(nx)
zb_wavy = -D .+ 0.4 .* sin.(2π .* x ./ L .* (nx-1) ./ (nx*5))
zb_realistic = -(
    10 .+ 20 .* (x ./ L) .+ 5 .* sin.(4π .* x ./ L) .+ 2 .* randn(nx)
)

# === Read Rhine bed profile from CSV (real data) ===
function read_rhine_profile(nx, D)
    try
        zb_rhine_file = readdlm("rhine_bed_profile.csv")
        zb_rhine = vec(zb_rhine_file)
        if length(zb_rhine) != nx
            @info "Resampling Rhine profile from $(length(zb_rhine)) to $nx points."
            oldx = range(0, 1, length=length(zb_rhine))
            newx = range(0, 1, length=nx)
            itp = interpolate((oldx,), zb_rhine, Gridded(Linear()))
            zb_rhine = itp.(newx)
        end
        return zb_rhine
    catch e
        @warn "Could not load 'rhine_bed_profile.csv', using flat bed instead. Error: $e"
        return -D * ones(nx)
    end
end

zb_rhine_real = read_rhine_profile(nx, D)

# --- CONFIG: Choose bed profile here ---
bed_profile = "rhine_real"  # options: "flat", "wavy", "realistic", "rhine_real"
zb = bed_profile == "rhine_real" ? zb_rhine_real :
     bed_profile == "wavy"      ? zb_wavy :
     bed_profile == "realistic" ? zb_realistic :
     zb_flat
output_prefix = bed_profile

# === Initial Conditions ===
h0 = 0.1 .* exp.(-100 .* ((x ./ L .- 0.5) .* L).^2) .- zb
q0 = zeros(nx)

# === State Vector Helpers ===
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

"""
    shallow_water_residual!(res, U, p)

Fills the residual vector `res` with the discretized shallow water equations for a given state `U` and parameter set `p`.
"""
function shallow_water_residual!(res, U, p)
    nx, dx, g, zb, cf = p.nx, p.dx, p.g, p.zb, p.cf
    h, q = unpack(U)
    T = eltype(U)

    dhdx = zeros(T, nx)
    dqdx = zeros(T, nx)
    dq2hdx = zeros(T, nx)
    dzetadx = zeros(T, nx)
    q_abs = abs.(q)

    left(i)  = i == 1    ? nx : i-1
    right(i) = i == nx   ? 1  : i+1

    for i in 1:nx
        dhdx[i]    = (h[right(i)] - h[left(i)]) / (2dx)
        dqdx[i]    = (q[right(i)] - q[left(i)]) / (2dx)
        dq2hdx[i]  = ((q[right(i)]^2 / h[right(i)]) - (q[left(i)]^2 / h[left(i)])) / (2dx)
        dzetadx[i] = ((h[right(i)] + zb[right(i)]) - (h[left(i)] + zb[left(i)])) / (2dx)
    end

    res[1:nx] = dqdx
    res[nx+1:2nx] = dq2hdx .+ g .* h .* dzetadx .+ cf .* q .* q_abs ./ (h.^2)
    return res
end

function timeloop(U0, params; dt=1e-3, tstart=0.0, tstop=1.0)
    nsteps = Int(round((tstop-tstart)/dt))
    U_hist = [U0]
    t_hist = [tstart]

    U_prev = copy(U0)
    t = tstart

    for step = 1:nsteps
        t += dt
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

# === Run the Simulation ===

U_hist, t_hist = timeloop(U0, params; dt=dt, tstart=tstart, tstop=tstop)

# === Post-Processing and Plotting ===

h_init, q_init = unpack(U_hist[1])
h_final, q_final = unpack(U_hist[end])

# Plot initial and final water depth
plt1 = plot(
    x, h_init, label="Initial h(x)", xlabel="x [km]", ylabel="Water Depth [m]",
    title="Water Depth: Initial vs Final", legend=:topright
)
plot!(plt1, x, h_final, label="Final h(x)", linestyle=:dash)
savefig(plt1, "$(output_prefix)_water_depth_comparison.png")

# Plot initial and final discharge
plt2 = plot(
    x, q_init, label="Initial q(x)", xlabel="x [km]", ylabel="Discharge [m²/s]",
    title="Discharge: Initial vs Final", legend=:topright
)
plot!(plt2, x, q_final, label="Final q(x)", linestyle=:dash)
savefig(plt2, "$(output_prefix)_discharge_comparison.png")

println("Plots saved as '$(output_prefix)_water_depth_comparison.png' and '$(output_prefix)_discharge_comparison.png'.")

# === Animation of Water Depth Evolution ===

frame_step = max(1, length(U_hist) ÷ 100)
frames = 1:frame_step:length(U_hist)

anim = @animate for idx in frames
    h, _ = unpack(U_hist[idx])
    plot(
        x, h,
        ylim=(minimum(h0) - 0.2, maximum(h0) + 0.2),
        xlabel="x [km]",
        ylabel="Water Depth h(x)",
        title = @sprintf("Water Depth Evolution at t = %.3f s", t_hist[idx]),
        legend=false
    )
end

gif(anim, "$(output_prefix)_water_depth_evolution.gif", fps=20)
println("Animation saved as '$(output_prefix)_water_depth_evolution.gif'.")

