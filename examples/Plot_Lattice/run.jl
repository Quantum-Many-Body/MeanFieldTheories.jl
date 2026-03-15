"""
Honeycomb lattice visualization using plot_lattice (Makie extension).

Draws a 4×4 tiled honeycomb lattice with NN bonds (red) and NNN bonds (blue),
sites colored by sublattice (A=green, B=purple).

Run:
    julia --project=examples examples/Plot_Lattice/run.jl
"""

using MeanFieldTheories
using CairoMakie

# ── Honeycomb unit cell ────────────────────────────────────────────────────────
# Primitive vectors: a1=(0,√3), a2=(3/2,√3/2)
# Two sites per cell: A at (0,0), B at (1,0)
const fig_dir = joinpath(@__DIR__, "..", "..", "docs", "src", "fig")

const a1 = [0.0, sqrt(3.0)]
const a2 = [1.5, sqrt(3.0)/2]

unitcell = Lattice(
    [Dof(:cell, 1), Dof(:sub, 2, [:A, :B])],
    [QN(cell=1, sub=1), QN(cell=1, sub=2)],
    [[0.0, 0.0], [1.0, 0.0]];
    vectors = [a1, a2])

# ── Tile 4×4 ──────────────────────────────────────────────────────────────────
lattice = Lattice(unitcell, (4, 4))

# ── Bond lists ────────────────────────────────────────────────────────────────
nn_bonds  = bonds(lattice, (:p, :p), 1)
nnn_bonds = bonds(lattice, (:p, :p), 2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plot_lattice(lattice;
    bond_lists   = [nn_bonds, nnn_bonds],
    bond_colors  = [:tomato, :steelblue],
    bond_labels  = ["NN", "NNN"],
    bond_widths  = [1.5, 1.0],
    bond_styles  = [:solid, :dash],
    site_groupby = :sub,
    site_colors  = [:seagreen, :mediumpurple],
    title        = "Honeycomb lattice  (4×4)")
    
display(fig)
out = joinpath(fig_dir, "lattice_bonds.png")
save(out, fig)
println("\nSaved: $out")
