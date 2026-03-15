"""
Visual test of plot_magnetization for three representative magnetic orders.

Positions and bond pairs are derived directly from Lattice objects, not
hardcoded, to demonstrate the intended workflow.

Case 1 — Square lattice Néel AFM (pure z):
  2×2 magnetic unit cell (4 sites), staggered mz = ±0.45.
  Expected: only ⊙/⊗ markers, no arrows.

Case 2 — Triangular lattice 120° Néel order (pure in-plane):
  √3×√3 magnetic unit cell (3 sites, one per sublattice).
  Note: 3 sublattices cannot fit evenly into a 2×2 = 4-site cell.
  Expected: only arrows, no ⊙/⊗.

Case 3 — Triangular lattice canted 120° Néel under uniform field (xyz):
  Same 3-site unit cell as Case 2, but a uniform magnetic field along +z
  cants all spins by θ = 55° from z, while preserving the 120° in-plane
  arrangement. All sites show ⊙ + arrows simultaneously.

Run:
    julia --project=examples examples/Plot_magnetization/run.jl
"""

using MeanFieldTheories
using CairoMakie

const fig_dir = joinpath(@__DIR__, "..", "..", "docs", "src", "fig")

# ── Helper: (i,j) pairs for visualization from a bonds list ───────────────────
# Maps each Bond's states to their indices in lattice.position_states.
# Inter-cell bonds (periodic images) are included as intra-cell pairs so they
# show as direct connections between unit-cell sites in the plot.
function bond_pairs(bond_list, lattice)
    state_idx = Dict(s => i for (i, s) in enumerate(lattice.position_states))
    seen = Set{Tuple{Int,Int}}()
    for b in bond_list
        length(b.states) == 2 || continue
        i = get(state_idx, b.states[1], nothing)
        j = get(state_idx, b.states[2], nothing)
        (isnothing(i) || isnothing(j)) && continue
        push!(seen, (min(i, j), max(i, j)))
    end
    return sort!(collect(seen))
end

# ── Helper: build a mag NamedTuple from (site_index, n, mx, my, mz) ───────────
function make_mag(site, n, mx, my, mz)
    m     = sqrt(mx^2 + my^2 + mz^2)
    theta = m > 1e-10 ? atan(sqrt(mx^2 + my^2), mz) : 0.0
    phi   = (mx^2 + my^2) > 1e-20 ? atan(my, mx) : 0.0
    (label     = (site = site,),
     n         = n,
     mx        = mx,
     my        = my,
     mz        = mz,
     m         = m,
     theta_deg = rad2deg(theta),
     phi_deg   = rad2deg(phi))
end

# ── Case 1: Square lattice Néel AFM ───────────────────────────────────────────
# Magnetic unit cell: 2×2 square, primitive vectors [2,0] and [0,2].
# Sublattice Q=(π,π) phases: (+,−,−,+) for sites at (0,0),(1,0),(0,1),(1,1).
sq_uc = Lattice(
    [Dof(:site, 4)],
    [QN(site=i) for i in 1:4],
    [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]];
    vectors = [[2.0,0.0],[0.0,2.0]])

sq_NN    = bonds(sq_uc, (:p,:p), 1)
sq_bonds = bond_pairs(sq_NN, sq_uc)
sq_pos   = sq_uc.coordinates   # Vector{SVector{2,Float64}}

mz0 = 0.45
afm_mags = [
    make_mag(1, 1.0,  0.0, 0.0, +mz0),   # ⊙
    make_mag(2, 1.0,  0.0, 0.0, -mz0),   # ⊗
    make_mag(3, 1.0,  0.0, 0.0, -mz0),   # ⊗
    make_mag(4, 1.0,  0.0, 0.0, +mz0),   # ⊙
]

println("Case 1 — Square lattice Néel AFM (pure z):")
print_magnetization(afm_mags)

fig1 = plot_magnetization(afm_mags, sq_pos;
           title = "Square-lattice Néel AFM  (pure z)",
           bonds = sq_bonds)
save(joinpath(fig_dir, "case1_square_afm_z.png"), fig1)
println("Saved case1_square_afm_z.png\n")

# ── Case 2 & 3 shared geometry: triangular √3×√3 magnetic unit cell ───────────
# Primitive triangular vectors: a1=(1,0), a2=(1/2,√3/2).
# √3×√3 supercell vectors: A1 = a1+a2 = (3/2,√3/2), A2 = (0,√3).
# 3 sites (one per sublattice); this is the minimum cell for the 120° order.
tri_uc = Lattice(
    [Dof(:site, 3)],
    [QN(site=i) for i in 1:3],
    [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3)/2]];
    vectors = [[3/2, sqrt(3)/2], [0.0, sqrt(3)]])

tri_NN    = bonds(tri_uc, (:p,:p), 1)
tri_bonds = bond_pairs(tri_NN, tri_uc)
tri_pos   = tri_uc.coordinates

# ── Case 2: Triangular lattice 120° Néel order ────────────────────────────────
# User-specified directions (clockwise by 120° each step):
#   site 1 (0,0)      → 90°   : (0,  m0, 0)
#   site 2 (1,0)      → −150° : (−√3/2·m0, −1/2·m0, 0)
#   site 3 (0.5,√3/2) → −30°  : (+√3/2·m0, −1/2·m0, 0)
m0 = 0.45
tri_mags = [
    make_mag(1, 1.0,  0.0,             +m0,     0.0),   # 90°
    make_mag(2, 1.0, -m0*sqrt(3)/2,   -m0/2,   0.0),   # −150°
    make_mag(3, 1.0, +m0*sqrt(3)/2,   -m0/2,   0.0),   # −30°
]

println("Case 2 — Triangular lattice 120° Néel order (pure xy):")
print_magnetization(tri_mags)

fig2 = plot_magnetization(tri_mags, tri_pos;
           title        = "Triangular-lattice 120° Néel  (pure xy)",
           bonds        = tri_bonds,
           axis_padding = 0.4)
save(joinpath(fig_dir, "case2_triangular_120neel_xy.png"), fig2)
println("Saved case2_triangular_120neel_xy.png\n")

# ── Case 3: 120° Néel canted by a uniform +z magnetic field ───────────────────
# Applying a field along +z cants all spins uniformly toward +z while
# preserving the relative 120° in-plane arrangement.
# θ = 55° from z  →  mz = m0·cos55°,  m_xy = m0·sin55°
# All three sites have the same mz (same ⊙), with 120° arrows in-plane.
θ_cant = 55 * π / 180
mz_c   = m0 * cos(θ_cant)
mxy_c  = m0 * sin(θ_cant)

cant_mags = [
    make_mag(1, 1.0,  0.0,              +mxy_c,   mz_c),   # 90°  + ⊙
    make_mag(2, 1.0, -mxy_c*sqrt(3)/2,  -mxy_c/2,  mz_c),   # −150° + ⊙
    make_mag(3, 1.0, +mxy_c*sqrt(3)/2,  -mxy_c/2,  mz_c),   # −30°  + ⊙
]

println("Case 3 — 120° Néel canted by +z field (xyz):")
print_magnetization(cant_mags)

fig3 = plot_magnetization(cant_mags, tri_pos;
           title        = "Triangular-lattice canted 120° Néel  (field ∥ z)",
           bonds        = tri_bonds,
           axis_padding = 0.4)
save(joinpath(fig_dir, "case3_canted_120neel_xyz.png"), fig3)
println("Saved case3_canted_120neel_xyz.png")
