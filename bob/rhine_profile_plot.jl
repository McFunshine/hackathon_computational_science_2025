using DelimitedFiles
using Plots

"""
    plot_bed_profile(filename::String; output_png::String="bed_profile.png", rivername::String="Rhine at Duisburg-Grossenbaum")

Reads a bed profile from CSV and plots it versus normalized domain (0–1). 
Optionally, set output_png filename and rivername for labeling.
"""
function plot_bed_profile(filename::String; output_png::String="bed_profile.png", rivername::String="Rhine at Duisburg-Grossenbaum")
    zb = vec(readdlm(filename))
    nx = length(zb)
    x = LinRange(0, 1, nx)  # Normalized position across cross-section

    plt = plot(
        x, zb,
        xlabel = "Relative position across cross-section",
        ylabel = "Bed elevation (m)",
        title  = rivername,
        legend = false,
        lw = 2,
        grid = true
    )
    savefig(plt, output_png)
    println("Plot saved as $(output_png)")
end

# Example usage:
plot_bed_profile("rhine_bed_profile.csv", output_png="rhine_profile_plot.png")
