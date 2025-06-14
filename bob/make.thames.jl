using DelimitedFiles

# Example: make a fake Thames array here; in real use, replace with actual data!
zb_thames_array = -5 .- 2 .* sin.(range(0, 2π, length=200))  # <-- replace this with real data!

# Write as a single-column CSV file
writedlm("thames_bed_profile.csv", zb_thames_array)
