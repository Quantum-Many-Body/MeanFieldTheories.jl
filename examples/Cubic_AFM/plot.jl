"""
Read res.dat and regenerate the cubic AFM figure without rerunning the HF solver.

Run:
    julia --project=examples examples/Cubic_AFM/plot.jl
"""

using Printf
using CairoMakie

const fig_dir = joinpath(@__DIR__, "..", "..", "docs", "src", "fig")

const U_vals       = [4.0, 8.0, 12.0, 16.0]
const mz_threshold = 0.01

# ── Read res.dat ──────────────────────────────────────────────────────────────
results = Dict{Float64, Vector{@NamedTuple{T::Float64, mz::Float64, converged::Bool}}}(
    U => [] for U in U_vals)

open(joinpath(@__DIR__, "res.dat")) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        isempty(strip(line))  && continue
        parts = split(line)
        U   = parse(Float64, parts[1])
        T   = parse(Float64, parts[2])
        mz  = parse(Float64, parts[3])
        conv = parts[4] == "true"
        push!(results[U], (T=T, mz=mz, converged=conv))
    end
end

# rows within each U are in the order they were written (low → high)
for U in U_vals
    sort!(results[U], by = r -> r.T)
    println(@sprintf("U/t = %.0f   %d points", U, length(results[U])))
end

# ── Néel temperature ──────────────────────────────────────────────────────────
T_Neel = Dict{Float64, Float64}()
for U in U_vals
    T_neel = 0.0
    for row in results[U]   # low → high: last hit = highest T with mz > threshold
        row.mz > mz_threshold && (T_neel = row.T)
    end
    T_Neel[U] = T_neel
    println(@sprintf("U/t = %.0f   T_Néel/t ≈ %.2f", U, T_neel))
end

# ── Plot ──────────────────────────────────────────────────────────────────────
markers = [:circle, :rect, :utriangle, :diamond]
colors  = [:tomato, :forestgreen, :royalblue, :mediumpurple]

fig = Figure(size=(560, 720))

# Panel (a): mz vs T
ax_a = Axis(fig[1, 1];
    xlabel = "T/t", ylabel = "mz",
    title  = "Cubic Hubbard model at half-filling  (L = 12)",
    limits = ((-0.2, 7.2), (-0.02, 0.52)),
    xgridvisible = false, ygridvisible = false)

for (i, U) in enumerate(U_vals)
    Ts  = [row.T  for row in results[U]]
    mzs = [row.mz for row in results[U]]
    scatterlines!(ax_a, Ts, mzs;
        marker=markers[i], color=(colors[i], 0.85),
        linewidth=1.2, markersize=11, label="U/t = $(Int(U))")
end
axislegend(ax_a; position=:rt)

# Panel (b): T_Néel vs U
ax_b = Axis(fig[2, 1];
    xlabel = "U/t", ylabel = "T_Néel/t",
    limits = ((-0.5, 17.5), (-0.15, 4.65)),
    xgridvisible = false, ygridvisible = false)

scatterlines!(ax_b, [0.0; collect(U_vals)], [0.0; [T_Neel[U] for U in U_vals]];
    color=:tomato, linewidth=1.5, markersize=12)

text!(ax_b, 3.0, 3.8; text="PM",  fontsize=16, font=:bold, color=:gray40)
text!(ax_b, 10.0, 0.6; text="AFM", fontsize=16, font=:bold, color=:gray40)
display(fig)
out = joinpath(fig_dir, "cubic_afm.png")
save(out, fig)
println("\nSaved: $out")
