"""
Read res.dat and regenerate the SDW/CDW phase diagram without rerunning the HF solver.

Run:
    julia --project=examples examples/SDW_CDW/plot.jl
"""

using Printf
using CairoMakie

const fig_dir = joinpath(@__DIR__, "..", "..", "docs", "src", "fig")
const U_ext = 4.0

# ── Read res.dat ──────────────────────────────────────────────────────────────
Vs_list    = Float64[]
Sq_list    = Float64[]
Nq_list    = Float64[]
phase_list = String[]

open(joinpath(@__DIR__, "res.dat")) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        isempty(strip(line))  && continue
        parts = split(line)
        push!(Vs_list,    parse(Float64, parts[1]))
        push!(Sq_list,    parse(Float64, parts[2]))
        push!(Nq_list,    parse(Float64, parts[3]))
        push!(phase_list, parts[4])
    end
end
println("Read $(length(Vs_list)) data points from res.dat")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(size=(600, 400))
ax  = Axis(fig[1, 1];
    xlabel = "V",
    ylabel = "Order parameter",
    title  = "Extended Hubbard model  (t=1, U=$(U_ext), half-filling)",
    xgridvisible = false,
    ygridvisible = false)

scatterlines!(ax, Vs_list, Sq_list;
    label="S(π,π)  staggered magnetization",
    marker=:circle, color=:green, linewidth=2, markersize=9)
scatterlines!(ax, Vs_list, Nq_list;
    label="N(π,π)  staggered density",
    marker=:rect, color=:blue, linewidth=2, markersize=9)
vlines!(ax, [U_ext/4];
    label="V/U = 1/4  (Vc = $(U_ext/4))",
    linestyle=:dash, color=:gray, linewidth=1)
axislegend(ax; position=:lt)

display(fig)
save(joinpath(fig_dir, "sdw_cdw.png"), fig)
println("Saved: docs/src/fig/sdw_cdw.png")
