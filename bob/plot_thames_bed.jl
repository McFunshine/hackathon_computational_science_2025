# plot_thames_bed.jl
# Plot the Thames bed profile from CSV

using DelimitedFiles
using Plots

# Read the bed profile (assumed one value per line, 200 lines)
zb = vec(readdlm("thames_bed_profile.csv"))
nx = length(zb)
x = LinRange(0, 1, nx)  # Domain (normalized: 0 to 1)

plot(
    x, zb,
    xlabel="Relative position along domain",
    ylabel="Bed elevation (zb)",
    title="Thames Bed Profile (from CSV)",
    legend=false,
    lw=2,
    grid=true
)
savefig("thames_bed_profile_plot.png")
println("Plot saved as thames_bed_profile_plot.png")
