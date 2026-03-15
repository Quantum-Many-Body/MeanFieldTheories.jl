"""
Read data files produced by run.jl and regenerate all SM_AFM figures.

Data files expected in the same directory:
  bands_2d.dat   — U=0 graphene 2D band structure
  bands_cyl.dat  — U=0 zigzag cylinder band structure
  res.dat        — AFM order parameter vs U sweep
  afm_bands.dat  — mean-field band structure for selected U values

Run:
    julia --project=examples examples/SM_AFM/plot.jl
"""

using Printf
using LinearAlgebra
using CairoMakie

const fig_dir = joinpath(@__DIR__, "..", "..", "docs", "src", "fig")

const t  = 1.0
const Uc = 2.2
const Ny = 50

# ── Reciprocal lattice for k-path tick marks ──────────────────────────────────
const a  = 1.0
const a1 = [a, 0.0]
const a2 = [0.5*a, √3/2*a]
let B = 2π * inv(hcat(a1, a2))'
    global b1 = B[:, 1]
    global b2 = B[:, 2]
end
const Γ = [0.0, 0.0]
const K = (2b1 + b2) / 3
const M = (b1 + b2) / 2
const d_ΓK = norm(K - Γ)
const d_KM = norm(M - K)
const d_MΓ = norm(Γ - M)
const arc_total  = d_ΓK + d_KM + d_MΓ
const xtick_pos  = [0.0, d_ΓK, d_ΓK+d_KM, arc_total]
const xtick_lab  = ["Γ", "K", "M", "Γ"]

# ── Read bands_2d.dat ─────────────────────────────────────────────────────────
arc2d  = Float64[]
E2d    = [Float64[], Float64[]]

open(joinpath(@__DIR__, "bands_2d.dat")) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        isempty(strip(line))  && continue
        p = parse.(Float64, split(line))
        push!(arc2d, p[1]); push!(E2d[1], p[2]); push!(E2d[2], p[3])
    end
end

# ── Read bands_cyl.dat ────────────────────────────────────────────────────────
kx_vals  = Float64[]
Ecyl     = Vector{Float64}[]   # one vector per band, filled after first line

open(joinpath(@__DIR__, "bands_cyl.dat")) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        isempty(strip(line))  && continue
        p = parse.(Float64, split(line))
        if isempty(Ecyl)
            append!(Ecyl, [Float64[] for _ in 2:length(p)])
        end
        push!(kx_vals, p[1])
        for (n, v) in enumerate(p[2:end])
            push!(Ecyl[n], v)
        end
    end
end

# ── Read res.dat ──────────────────────────────────────────────────────────────
U_sweep_dat = Float64[]
mafm_dat    = Float64[]

open(joinpath(@__DIR__, "res.dat")) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        isempty(strip(line))  && continue
        p = split(line)
        push!(U_sweep_dat, parse(Float64, p[1]))
        push!(mafm_dat,    parse(Float64, p[2]))
    end
end

# ── Read afm_bands.dat ────────────────────────────────────────────────────────
afm_data = Dict{Float64, Tuple{Vector{Float64}, Matrix{Float64}}}()

open(joinpath(@__DIR__, "afm_bands.dat")) do f
    for line in eachline(f)
        startswith(line, "#") && continue
        isempty(strip(line))  && continue
        p  = parse.(Float64, split(line))
        U  = p[1]; a_val = p[2]; Es = p[3:6]
        if !haskey(afm_data, U)
            afm_data[U] = (Float64[], zeros(Float64, 4, 0))
        end
        arc_v, mat = afm_data[U]
        push!(arc_v, a_val)
        afm_data[U] = (arc_v, hcat(mat, Es))
    end
end
U_bands = sort(collect(keys(afm_data)))

# ── Figure 1: U=0 graphene 2D bands ──────────────────────────────────────────
fig0 = Figure(size=(600, 450))
ax0  = Axis(fig0[1, 1];
    title  = "Graphene band structure (U=0)",
    xlabel = "k-path", ylabel = "E / t",
    xticks = (xtick_pos, xtick_lab),
    limits = ((-0.05, arc_total+0.05), (-3.5, 3.5)),
    xgridvisible = false, ygridvisible = false)
hlines!(ax0, [0.0];     color=:gray, linestyle=:dash, linewidth=0.8)
vlines!(ax0, xtick_pos; color=:gray, linestyle=:dot,  linewidth=0.8)
for E in E2d
    lines!(ax0, arc2d, E; color=:steelblue, linewidth=1.5)
end
display(fig0)
save(joinpath(fig_dir, "graphene_bands_2d.png"), fig0)
println("Saved: docs/src/fig/graphene_bands_2d.png")

# ── Figure 2: Zigzag cylinder bands ──────────────────────────────────────────
cyl_xtick_pos = [-π, -2π/3, 0.0, 2π/3, π]
cyl_xtick_lab = ["-π", "-2π/3", "0", "2π/3", "π"]

fig1 = Figure(size=(600, 450))
ax1  = Axis(fig1[1, 1];
    title  = "Graphene zigzag cylinder (Ny=$Ny, U=0)",
    xlabel = "kx", ylabel = "E / t",
    xticks = (cyl_xtick_pos, cyl_xtick_lab),
    limits = ((-π, π), (-3.5, 3.5)),
    xgridvisible = false, ygridvisible = false)
hlines!(ax1, [0.0];          color=:gray, linestyle=:dash, linewidth=0.8)
vlines!(ax1, [-2π/3, 2π/3]; color=:gray, linestyle=:dot,  linewidth=0.8)
for E in Ecyl
    lines!(ax1, kx_vals, E; color=:steelblue, linewidth=0.6, alpha=0.7)
end
display(fig1)
save(joinpath(fig_dir, "graphene_bands_cylinder.png"), fig1)
println("Saved: docs/src/fig/graphene_bands_cylinder.png")

# ── Figure 3: AFM order parameter vs U ───────────────────────────────────────
fig_ord = Figure(size=(600, 400))
ax_ord  = Axis(fig_ord[1, 1];
    xlabel = "U / t", ylabel = "m_AF",
    title  = "Honeycomb Hubbard: AFM order at half-filling",
    xgridvisible = false, ygridvisible = false)
scatterlines!(ax_ord, U_sweep_dat, mafm_dat;
    color=:darkred, linewidth=2, markersize=9, label="m_AF (ground state)")
vlines!(ax_ord, [Uc]; label="Uc ≈ $(Uc)", linestyle=:dash, color=:gray, linewidth=1)
axislegend(ax_ord; position=:lt)
display(fig_ord)
save(joinpath(fig_dir, "afm_order_parameter.png"), fig_ord)
println("Saved: docs/src/fig/afm_order_parameter.png")

# ── Figure 4: Mean-field bands for selected U ─────────────────────────────────
nb   = length(U_bands)
rows = nb <= 3 ? 1 : 2
cols = ceil(Int, nb / rows)

fig_bands = Figure(size=(350*cols, 310*rows))
for (i, U) in enumerate(U_bands)
    row = div(i-1, cols) + 1
    col = mod(i-1, cols) + 1
    arc_v, mat = afm_data[U]
    ax = Axis(fig_bands[row, col];
        title  = "U/t = $(U)",
        xticks = (xtick_pos, xtick_lab),
        limits = ((-0.05, arc_total+0.05), (-4.0+U/2, 4.0+U/2)),
        xgridvisible = false, ygridvisible = false)
    hlines!(ax, [U/2];      color=:gray, linestyle=:dash, linewidth=0.8)
    vlines!(ax, xtick_pos;  color=:gray, linestyle=:dot,  linewidth=0.8)
    for n in 1:size(mat, 1)
        lines!(ax, arc_v, mat[n, :]; color=:steelblue, linewidth=1.2)
    end
end
display(fig_bands)
save(joinpath(fig_dir, "afm_bands.png"), fig_bands)
println("Saved: docs/src/fig/afm_bands.png")
